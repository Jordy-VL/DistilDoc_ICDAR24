import os
import numpy as np
from datasets import load_dataset, Features, ClassLabel
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)


def split_balanced_samples(datasplit, N_K=25, label_column="label"):
    datasplit = datasplit.shuffle(seed=42, keep_in_memory=False)

    idx = {int(j): [] for j in datasplit.unique(label_column)}
    for i, x in enumerate(datasplit):
        if len(idx[x[label_column]]) < N_K:
            idx[x[label_column]].append(i)
        if all([len(idx[k]) == N_K for k, v in idx.items()]):
            break
    all_indices = []
    for k, v in idx.items():
        all_indices.extend(v)
    datasplit = datasplit.select(all_indices)
    return datasplit


def process_label_ids(batch, remapper, label_column="label"):
    batch[label_column] = [remapper[label_id] for label_id in batch[label_column]]
    return batch


def build_dataset(config, split, processor=None, add_indices=False):
    # config["dataset"] = config["dataset"].lower()

    if len(config["dataset"].split("/")) > 2:  # imagefolder
        data = load_dataset("imagefolder", data_dir=config["dataset"], split=split)

    elif config["dataset"] == "jordyvl/rvl_cdip_100_examples_per_class":
        data = load_dataset(config["dataset"], split=split)

    elif config["dataset"] == "jordyvl/RVL-CDIP-N":
        data = load_dataset(config["dataset"], split="test")
        label2idx = {label.replace(" ", "_"): i for label, i in config["label2id"].items()}
        data_label2idx = {label: i for i, label in enumerate(data.features["label"].names)}
        remapper = {}
        for k, v in label2idx.items():
            if k in data_label2idx:
                remapper[data_label2idx[k]] = v
        new_features = Features(
            {
                **{k: v for k, v in data.features.items() if k != "label"},
                "label": ClassLabel(num_classes=len(label2idx), names=list(label2idx.keys())),
            }
        )
        data = data.map(
            lambda example: process_label_ids(example, remapper),
            features=new_features,
            batched=True,
            batch_size=100,
            desc="Aligning the labels",
        )
    elif config["dataset"] == "rvl_cdip":
        cache_dir = "/mnt/lerna/data/HFcache"
        data = load_dataset(config["dataset"], split=split, cache_dir=cache_dir if os.path.exists(cache_dir) else None)

        if split == "test":
            data = data.select([i for i in range(len(data)) if i != 33669])  # corrupt

    elif "rvl_cdip" in config["dataset"]:
        cache_dir = "/mnt/lerna/data/HFcache"
        data = load_dataset("rvl_cdip", split=split, cache_dir=cache_dir if os.path.exists(cache_dir) else None)
        if split == "test":
            data = data.select([i for i in range(len(data)) if i != 33669])  # picklable
        else:
            # label_distribution = dict(zip(*np.unique(data['label'], return_counts=True))) #balanced anyway
            N_K = int(config["dataset"].lower().replace("nk", "").replace("_balanced", "").split("-")[-1])
            if split == "validation":
                N_K //= 4
            # sample N_k examples per class
            data = split_balanced_samples(data, N_K=N_K)

    elif config["dataset"] == "maveriq/tobacco3482":
        """
        In our setting, we random sampled three subsets to be used for train,
        validation and test, fixing their cardinality to 800, 200, and 2482 respectively, as in https://arxiv.org/abs/1907.06370
        """
        data = load_dataset(config["dataset"])
        train_size, val_size, test_size = 800, 200, 2482  # 80 examples per class for training

        # Perform the train-validation-test split
        shuffled = data["train"].shuffle(seed=42)
        train_dataset = shuffled.select(range(train_size))
        val_dataset = shuffled.select(range(train_size, train_size + val_size))
        test_dataset = shuffled.select(range(train_size + val_size, train_size + val_size + test_size))

        if split == "train":
            data = train_dataset
        elif split == "validation":
            data = val_dataset
        elif split == "test":
            data = test_dataset

    data = data.rename_column("label", "labels")

    # apply processor with speed-up
    batch_size = 1000
    num_proc = 1
    if os.cpu_count() > 30:
        num_proc = 10
        batch_size = 50
    elif os.cpu_count() > 10:  # should be new default locally
        num_proc = 4
        batch_size = 50

    encoded_dataset = data.map(
        lambda examples: processor([image.convert("RGB") for image in examples["image"]]),
        batched=True,
        num_proc=num_proc,
        batch_size=batch_size,
    )
    columns = ["pixel_values", "labels"]
    if add_indices:  # was wrong
        encoded_dataset = encoded_dataset.map(
            lambda examples: {"indices": list(range(len(examples["labels"])))},
            batched=True,
            batch_size=len(encoded_dataset),
            num_proc=1,
            remove_columns=["image"],
        )
        columns.append("indices")

    encoded_dataset.set_format(type="torch", columns=columns)

    # TODO: should we not apply some transforms as well?

    return encoded_dataset


def transformations(processor):
    normalize = Normalize(mean=processor.image_mean, std=processor.image_std)
    size = (
        processor.size["shortest_edge"]
        if "shortest_edge" in processor.size
        else (processor.size["height"], processor.size["width"])
    )
    _transforms = Compose([RandomResizedCrop(size), ToTensor(), normalize])
    return _transforms


def transforms(examples):
    examples["pixel_values"] = [transformations(img.convert("RGB")) for img in examples["image"]]
    del examples["image"]
    return examples


# to apply: .with_transform(transforms)
