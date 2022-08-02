import argparse
import numpy as np
import random
import torch
import torch.utils.data
import torch.nn.functional as F
import clip
import clip.model

from data.CoOpBenchmark import CoOpBenchmark


def main(args):
    model, aug = clip.load('ViT-B/32', device='cuda:' + str(args.gpu))
    # set the max text length to 10
    model.context_length = args.text_length
    model.positional_embedding.data = model.positional_embedding.data[:args.text_length]
    for layer in model.transformer.resblocks:
        layer.attn_mask.data = layer.attn_mask.data[:args.text_length, :args.text_length]

    # add projector
    model.t2i = torch.nn.Linear(512, 768, bias=False).cuda(args.gpu)

    # prepare training and testing dataloader
    datasets = CoOpBenchmark(args, aug)
    train_loader = torch.utils.data.DataLoader(datasets.train_dataset, batch_size=len(datasets.train_dataset), num_workers=1)
    test_loader = torch.utils.data.DataLoader(datasets.test_dataset, batch_size=128, num_workers=8, drop_last=False)

    text_features = get_text_feature(model, datasets.classnames, args)

    if args.shot >= 1:
        prototypes = train(model, train_loader, text_features, args)
    else:
        prototypes = None

    test(model, test_loader, prototypes, text_features, args)


def get_text_feature(teacher, classnames, args):
    if args.dataset == 'ucf101':
        text = ['A photo of a person doing ' + classname for classname in classnames]
    else:
        text = ['A photo of ' + classname for classname in classnames]
    if args.dataset == 'flowers102':
        for i in range(len(text)):
            text[i] = text[i] + ', a type of flower'
    text_token = clip.tokenize(text).cuda(args.gpu)
    if args.text_length != -1:
        text_token = text_token[:, :args.text_length]

    teacher.eval()
    with torch.no_grad():
        text_feature = teacher.encode_text(text_token)
        text_feature = text_feature.float()

    return text_feature

def train(model, train_loader, text_features, args):
    model.eval()
    optim = torch.optim.Adam(model.t2i.parameters(), lr=1e-3)
    images, labels, _ = next(iter(train_loader))  # The batch_size is equal to the dataset size
    images, labels = images.cuda(args.gpu), labels.cuda(args.gpu)
    labels, idx = torch.sort(labels)
    images = images[idx]
    with torch.no_grad():
        image_features = model.encode_image(images).float()
        image_features = F.normalize(image_features, dim=-1)

    for epoch in range(args.epochs):
        # If the shot is too large, we split the training data into batches.
        n_way, n_shot = text_features.shape[0], args.shot
        idxs = torch.stack([torch.randperm(n_shot)+i*n_shot for i in range(n_way)]).cuda(args.gpu)
        idxs = torch.split(idxs, args.max_shot, 1)
        n_step = int(np.ceil(n_shot/args.max_shot))
        for step in range(n_step):
            idx = idxs[step].contiguous().view(-1)
            image = images[idx]
            label = labels[idx]
            prompt = model.t2i(text_features)
            prompt = prompt[label]
            prototype = forward_with_svp(model.visual, image, prompt, args)
            prototype = prototype.view(n_way, -1, prototype.shape[-1]).mean(1)
            prototype = F.normalize(prototype, dim=-1)

            sim = image_features @ prototype.t()
            loss = F.cross_entropy(sim / args.t, labels)
            _, pred = sim.max(dim=-1)
            acc = pred.eq(labels).sum() / pred.shape[0]

            # The text embedding of a class name is also regarded as a training sample
            sim = F.normalize(text_features, dim=-1) @ prototype.t()
            loss_text = F.cross_entropy(sim / args.t, torch.arange(n_way).cuda(args.gpu))
            adaptive_loss_weight = n_shot / args.max_shot
            loss = adaptive_loss_weight * loss + args.lw_text * loss_text

            optim.zero_grad()
            loss.backward()
            optim.step()

    with torch.no_grad():
        prompt = model.t2i(text_features)
        prompt = prompt[labels]
        prototype = forward_with_svp(model.visual, images, prompt, args)
        prototype = prototype.view(text_features.shape[0], args.shot, -1).mean(1)

    return prototype


def test(model, test_loader, prototypes, text_feautres, args):
    model.eval()

    if prototypes is None:
        # zero-shot
        prototypes = F.normalize(text_feautres, dim=-1)
    else:
        # few-shot
        prototypes = F.normalize(prototypes, dim=-1)

    n_true = 0
    n_total = 0
    with torch.no_grad():
        for image, label, text in test_loader:
            image = image.cuda(args.gpu)
            label = label.cuda(args.gpu)

            image_feature = model.encode_image(image).float()

            sim = F.normalize(image_feature, dim=-1) @ prototypes.t()
            _, pred = sim.max(-1)
            n_true += label.eq(pred).sum().float().item()
            n_total += label.shape[0]

    acc = n_true/n_total
    print(f'Test acc: {acc*100: .2f}')


def forward_with_svp(self, x, text, args):
    # encode image with svp
    with torch.no_grad():
        x = x.type(self.proj.dtype)
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x],
            dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
    for i, layer in enumerate(self.transformer.resblocks):
        if i == args.svp_layer:
            x = torch.cat([x, text.unsqueeze(0).type(x.dtype)], dim=0)
        if i < args.svp_layer:
            with torch.no_grad():
                x = layer(x)
        else:
            x = layer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD

    x = self.ln_post(x[:, 0, :])

    if self.proj is not None:
        x = x @ self.proj

    return x.float()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=2)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='ucf101', choices=['caltech101', 'stanfordcars', 'flowers102', 'ucf101'])
    parser.add_argument('--shot', type=int, default=16)
    parser.add_argument('--max_shot', type=int, default=8)
    parser.add_argument('--lw_text', type=float, default=1.)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--text_length', type=int, default=20)
    parser.add_argument('--svp_layer', type=int, default=10)
    parser.add_argument('--t', type=float, default=0.1)
    parser.add_argument('--print_step', type=int, default=100)

    args = parser.parse_args()

    if args.seed >= 0:
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True

    main(args)