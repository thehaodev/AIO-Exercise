import os
import torch
import numpy as np
from torch.utils.data import DataLoader


def read_file(file_path):
    root_dir = file_path
    classes = {
        label_idx: class_name for label_idx, class_name in enumerate(
            sorted(os.listdir(root_dir))
        )
    }

    img_paths = []
    labels = []
    for label_idx, class_name in classes.items():
        class_dir = os.path.join(root_dir, class_name)
        for img_filename in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_filename)
            img_paths.append(img_path)
            labels.append(label_idx)

    return img_paths, labels, classes


def transform(img, img_size=(224, 224)):
    img = img.resize(img_size)
    img = np.array(img)[..., :3]
    img = torch.tensor(img).permute(2, 0, 1).float()
    normalized_img = img / 255.0
    return normalized_img


def data_loader(train_dataset, val_dataset, test_dataset, train_batch_size, test_batch_size):

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=test_batch_size,
        shuffle=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False
    )

    return train_loader, val_loader, test_loader
