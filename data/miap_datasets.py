import os
import json
import random
import shutil

from typing import List
import cv2
from tqdm import tqdm
import argparse
from combine_datasets import annotations_from_images


random.seed(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--percentage", type=float, default=1.0, help="Data percentage")
    opt = parser.parse_args()

    percentage = opt.percentage

    miap = json.load(open("data/miap/instances_all_miap.json", "r"))

    miap_train, miap_val = {
                               "images": list(),
                               "annotations": list()
                           }, {
                               "images": list(),
                               "annotations": list()
                           }

    miap_train["images"] = list()
    miap_val["images"] = list()

    miap_imgs = miap["images"][:int(percentage * len(miap["images"]))]

    print("MIAP number of images:", len(miap_imgs))

    random.shuffle(miap_imgs)

    val_ratio = 0.3

    miap_train["images"] = miap_imgs[:int((1 - val_ratio) * len(miap_imgs))]
    miap_val["images"] = miap_imgs[int((1 - val_ratio) * len(miap_imgs)):]

    print(len(miap_train["images"]), len(miap_val["images"]))

    miap_train["annotations"] = annotations_from_images(miap["annotations"], miap_train["images"], "miap_train")
    miap_val["annotations"] = annotations_from_images(miap["annotations"], miap_val["images"], "miap_val")

    train = {
        "images": miap_train["images"],
        "annotations": miap_train["annotations"],
        "categories": miap["categories"]
    }

    val = {
        "images": miap_val["images"],
        "annotations": miap_val["annotations"],
        "categories": miap["categories"]
    }

    print(val["categories"])

    if os.path.exists("data/images"):
        shutil.rmtree("data/images")

    os.mkdir("data/images")
    os.mkdir("data/images/val")
    os.mkdir("data/images/train")

    for img in val["images"]:
        filename = img["file_name"]
        shutil.copy(os.path.join("data", "miap", "images", filename),
                    os.path.join("data", "images", "val", filename))

    for img in train["images"]:
        filename = img["file_name"]
        shutil.copy(os.path.join("data", "miap", "images", filename),
                    os.path.join("data", "images", "train", filename))

    json.dump(train, open("data/combined_train.json", "w"))
    json.dump(val, open("data/combined_val.json", "w"))

    if os.path.exists("data/labels"):
        shutil.rmtree("data/labels")

    os.mkdir("data/labels")
    os.mkdir("data/labels/val")
    os.mkdir("data/labels/train")

    val_image_annotations = {}
    train_image_annotations = {}

    for ann in val["annotations"]:
        val_image_annotations[ann["image_id"]] = []

    for ann in val["annotations"]:
        t = list()
        t.append(ann["category_id"])
        t.extend(ann["bbox"])
        val_image_annotations[ann["image_id"]].append(t)

    for ann in train["annotations"]:
        train_image_annotations[ann["image_id"]] = []

    for ann in train["annotations"]:
        t = list()
        t.append(ann["category_id"])
        t.extend(ann["bbox"])
        train_image_annotations[ann["image_id"]].append(t)

    for idx, bboxes in val_image_annotations.items():
        if type(idx) != str and idx < 1_000_000:
            idx = "0" + str(idx)
        im = cv2.imread(f"data/images/val/{idx}.jpg")
        with open(f"data/labels/val/{idx}.txt", "w") as f:
            for box in bboxes:
                box[0] = 0
                box[1] /= im.shape[1]
                box[2] /= im.shape[0]
                box[3] /= im.shape[1]
                box[4] /= im.shape[0]
                box = [str(i) for i in box]
                box_str = " ".join(box)
                box_str += "\n"
                f.write(box_str)

    for idx, bboxes in train_image_annotations.items():
        if type(idx) != str and idx < 1_000_000:
            idx = "0" + str(idx)
        im = cv2.imread(f"data/images/train/{idx}.jpg")
        with open(f"data/labels/train/{idx}.txt", "w") as f:
            for box in bboxes:
                box[0] = 0
                box[1] /= im.shape[1]
                box[2] /= im.shape[0]
                box[3] /= im.shape[1]
                box[4] /= im.shape[0]
                box = [str(i) for i in box]
                box_str = " ".join(box)
                box_str += "\n"
                f.write(box_str)

    with open("data/train.txt", "w") as f:
        file_names = os.listdir("data/images/train")
        file_paths = [os.path.join("./data/images/train/", filename) for filename in file_names]
        for filepath in file_paths:
            f.write(filepath + "\n")

    with open("data/val.txt", "w") as f:
        file_names = os.listdir("data/images/val")
        file_paths = [os.path.join("./data/images/val/", filename) for filename in file_names]
        for filepath in file_paths:
            f.write(filepath + "\n")
