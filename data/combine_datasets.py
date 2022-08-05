import os
import json
import random
import shutil

from typing import List
import cv2
from tqdm import tqdm

import argparse

random.seed(1)


def annotations_from_images(annotations, images, name) -> List:
    new_annotations = list()

    image_ids = [i["id"] for i in images]

    for _, annotation in tqdm(enumerate(annotations), total=len(annotations), desc=name):
        if annotation["image_id"] in image_ids:
            if "segmentation" in annotation:
                del annotation["segmentation"]
                del annotation["iscrowd"]
                del annotation["area"]
            new_annotations.append(annotation)

    return new_annotations


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--percentage', type=float, default=1.0, help='data percentage')
    opt = parser.parse_args()

    percentage = opt.percentage

    miap = json.load(open("data/miap/instances_all_miap.json", "r"))
    modanet = json.load(open("data/modanet/instances_all_modanet_transformed.json", "r"))

    # We do not want all categories in modanet, therefore we remove the following categories

    remove_categories = ["sunglasses", "belt", "scarf/tie"]

    remove_category_ids = list()

    print("len of annotations before removal:", len(modanet["annotations"]))

    for idx, category in enumerate(modanet["categories"]):
        if category["name"] in remove_categories:
            remove_category_ids.append(category["id"])
            modanet["categories"].remove(category)

    new_annotations = list()

    for idx, annotation in tqdm(enumerate(modanet["annotations"]), total=len(modanet["annotations"]),
                                desc="Removing annotations with removed categories"):
        if not annotation["category_id"] in remove_category_ids:
            new_annotations.append(annotation)

    modanet["annotations"] = new_annotations

    miap_train, miap_val = {
                               "images": [],
                               "annotations": []
                           }, {
                               "images": [],
                               "annotations": []
                           }

    modanet_train, modanet_val = {
                                     "images": [],
                                     "annotations": []
                                 }, {
                                     "images": [],
                                     "annotations": []
                                 }

    miap_train["images"] = []
    modanet_train["images"] = []

    miap_val["images"] = []
    modanet_val["images"] = []

    miap_imgs = miap["images"][:int(percentage * len(miap["images"]))]
    modanet_imgs = modanet["images"][:int(percentage * len(modanet["images"]))]

    print("MIAP number of images:", len(miap_imgs))
    print("Modanet number of images:", len(modanet_imgs))

    random.shuffle(miap_imgs)
    random.shuffle(modanet_imgs)

    val_ratio = 0.1

    miap_train["images"] = miap_imgs[:int((1 - val_ratio) * len(miap_imgs))]
    miap_val["images"] = miap_imgs[int((1 - val_ratio) * len(miap_imgs)):]

    modanet_train["images"] = modanet_imgs[:int((1 - val_ratio) * len(modanet_imgs))]
    modanet_val["images"] = modanet_imgs[int((1 - val_ratio) * len(modanet_imgs)):]

    print(len(miap_train["images"]), len(miap_val["images"]), len(modanet_train["images"]), len(modanet_val["images"]))

    highest_category_id = max([i["id"] for i in modanet["categories"]])
    print("highest_category_id:", highest_category_id)

    # Adjust the ids of the second dataset

    for xid, category in enumerate(miap["categories"]):
        miap["categories"][xid]["id"] += highest_category_id

    for xid, annotation in enumerate(miap["annotations"]):
        miap["annotations"][xid]["category_id"] += highest_category_id

    modanet_train["annotations"] = annotations_from_images(modanet["annotations"], modanet_train["images"],
                                                           "modanet_train")
    modanet_val["annotations"] = annotations_from_images(modanet["annotations"], modanet_val["images"], "modanet_val")
    miap_train["annotations"] = annotations_from_images(miap["annotations"], miap_train["images"], "miap_train")
    miap_val["annotations"] = annotations_from_images(miap["annotations"], miap_val["images"], "miap_val")

    train = {
        "images": modanet_train["images"] + miap_train["images"],
        "annotations": modanet_train["annotations"] + miap_train["annotations"],
        "categories": modanet["categories"] + miap["categories"]
    }

    val = {
        "images": modanet_val["images"] + miap_val["images"],
        "annotations": modanet_val["annotations"] + miap_val["annotations"],
        "categories": modanet["categories"] + miap["categories"]
    }

    new_category_ids = {}

    for idx, category in enumerate(val["categories"]):
        new_category_ids[category["id"]] = idx

    cats = []

    for cat in val["categories"]:
        cats.append(cat["id"])

    train_cats = []
    train_anns = []

    for annotation in train["annotations"]:
        ann = annotation
        ann["category_id"] = new_category_ids[annotation["category_id"]]
        train_anns.append(ann)

    train["annotations"] = train_anns

    for category in train["categories"]:
        cat = category
        cat["id"] = new_category_ids[category["id"]]
        train_cats.append(cat)

    train["categories"] = train_cats

    val_cats = []
    val_anns = []

    for annotation in val["annotations"]:
        ann = annotation
        ann["category_id"] = new_category_ids[annotation["category_id"]]
        val_anns.append(ann)

    val["annotations"] = val_anns

    cats = []
    for i in train["annotations"]:
        cats.append(i["category_id"])

    if os.path.exists("data/images"):
        shutil.rmtree("data/images")

    os.mkdir("data/images")
    os.mkdir("data/images/val")
    os.mkdir("data/images/train")

    for img in val["images"]:
        filename = img["file_name"]
        try:
            shutil.copy(os.path.join("data", "modanet", "images", filename),
                        os.path.join("data", "images", "val", filename))
        except IOError as e:
            shutil.copy(os.path.join("data", "miap", "images", filename),
                        os.path.join("data", "images", "val", filename))

    for img in train["images"]:
        filename = img["file_name"]
        try:
            shutil.copy(os.path.join("data", "modanet", "images", filename),
                        os.path.join("data", "images", "train", filename))
        except IOError as e:
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
        print(str(ann["image_id"]))
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
