import json
import os
import pickle
import random
from collections import defaultdict
from PIL import Image


dataset_path = {
    'caltech101': '../dataset/Caltech101/',
    'stanfordcars': '../dataset/StanfordCars/',
    'flowers102': '../dataset/Flowers102/',
    'ucf101': '../dataset/UCF101',
}
image_path = {
    'caltech101': '101_ObjectCategories',
    'stanfordcars': '',
    'flowers102': 'jpg',
    'ucf101': 'UCF-101-midframes',
}
split_path = {
    'caltech101': 'split_zhou_Caltech101.json',
    'stanfordcars': 'split_zhou_StanfordCars.json',
    'flowers102': 'split_zhou_OxfordFlowers.json',
    'ucf101': 'split_zhou_UCF101.json',
}


# Acknowledgement
# We refer to this project for implementation: https://github.com/KaiyangZhou/CoOp?utm_source=catalyzex.com
class CoOpBenchmark(object):
    def __init__(self, args, transform):
        self.args = args
        self.dataset_dir = dataset_path[args.dataset]
        self.image_dir = os.path.join(self.dataset_dir, image_path[args.dataset])
        self.split_path = os.path.join(self.dataset_dir, split_path[args.dataset])
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        os.makedirs(self.split_fewshot_dir, exist_ok=True)

        train, val, test, classnames = read_split(self.split_path, self.image_dir)
        self.classnames = classnames

        num_shots = args.shot
        if num_shots >= 1:
            seed = args.seed
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")

            if os.path.exists(preprocessed):
                # print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train, val = data["train"], data["val"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
                data = {"train": train, "val": val}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        self.train_dataset = DatasetWithClassName(train, transform)
        self.val_dataset = DatasetWithClassName(val, transform)
        self.test_dataset = DatasetWithClassName(test, transform)

    def generate_fewshot_dataset(self, *data_sources, num_shots=-1, repeat=False):
        """Generate a few-shot dataset (typically for the training set).

        This function is useful when one wants to evaluate a model
        in a few-shot learning setting where each class only contains
        a small number of images.

        Args:
            data_sources: each individual is a list containing Datum objects.
            num_shots (int): number of instances per class to sample.
            repeat (bool): repeat images if needed (default: False).
        """
        if num_shots < 1:
            if len(data_sources) == 1:
                return data_sources[0]
            return data_sources

        print(f"Creating a {num_shots}-shot dataset")
        random.seed(self.args.seed)

        output = []

        for data_source in data_sources:
            tracker = self.split_dataset_by_label(data_source)
            dataset = []

            for label, items in tracker.items():
                if len(items) >= num_shots:
                    sampled_items = random.sample(items, num_shots)
                else:
                    if repeat:
                        sampled_items = random.choices(items, k=num_shots)
                    else:
                        sampled_items = items
                dataset.extend(sampled_items)

            output.append(dataset)

        random.seed(self.args.seed)

        if len(output) == 1:
            return output[0]

        return output

    def split_dataset_by_label(self, data_source):
        """Split a dataset, i.e. a list of Datum objects,
        into class-specific groups stored in a dictionary.

        Args:
            data_source (list): a list of Datum objects.
        """
        output = defaultdict(list)

        for item in data_source:
            output[item['label']].append(item)

        return output


def read_split(filepath, path_prefix):
    def _convert(items):
        out = []
        label2name = dict()
        for impath, label, classname in items:
            impath = os.path.join(path_prefix, impath)
            # The class name "water_lilly" in Caltch101 is wrong!
            if classname == 'water_lilly':
                classname = 'water_lily'
            classname = classname.replace('_', ' ')
            item = dict(impath=impath, label=int(label), classname=classname)
            out.append(item)
            if int(label) not in label2name:
                label2name[label] = classname
        return out, label2name

    # print(f"Reading split from {filepath}")
    split = read_json(filepath)
    train, label2name = _convert(split["train"])
    val, _ = _convert(split["val"])
    test, _ = _convert(split["test"])
    classnames = [label2name[i] for i in range(len(label2name))]

    return train, val, test, classnames


def read_json(fpath):
    """Read json file from a path."""
    with open(fpath, "r") as f:
        obj = json.load(f)
    return obj


class DatasetWithClassName(object):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, i):
        item = self.dataset[i]
        with open(item['impath'], 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        img = self.transform(img)
        label = item['label']
        return img, label, item['classname']

    def __len__(self):
        return len(self.dataset)