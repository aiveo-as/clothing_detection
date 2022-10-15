import os
import json
import random
import shutil

from typing import List
import cv2
from tqdm import tqdm

import argparse

from combine_datasets import annotations_from_images

#random.seed(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--percentage', type=float, default=1.0, help='data percentage')
    opt = parser.parse_args()

    percentage = opt.percentage


    modanet = json.load(open("data/modanet/instances_all_modanet_transformed.json", "r"))
    max_cat_id = max([i["id"] for i in modanet["categories"]])

    for idx, img in enumerate(modanet["images"]):
        modanet["images"][idx]["id"] = int(modanet["images"][idx]["id"])

    for idx, ann in enumerate(modanet["annotations"]):
        modanet["annotations"][idx]["image_id"] = int(modanet["annotations"][idx]["image_id"])

    modanet["categories"].append({"supercategory": "fashion", "id": max_cat_id + 1, "name": "person"})

    val_predicted_humans = json.load(open("data/modanet/val_person_best_predictions.json", "r"))
    
    for idx, i in enumerate(val_predicted_humans):
        bb = i["bbox"]
        bb = [bb[0], bb[1], bb[2], bb[3]]
        val_predicted_humans[idx]["bbox"] = bb
        val_predicted_humans[idx]["category_id"] = max_cat_id + 1

    image_id_to_image_filename = {}


    for i in modanet["images"]:
        image_id_to_image_filename[int(i["id"])] = i["file_name"]

    train_predicted_humans = json.load(open("data/modanet/train_person_best_predictions.json", "r"))

    for idx, i in enumerate(train_predicted_humans):
        bb = i["bbox"]
        bb = [bb[0], bb[1], bb[2], bb[3]]
        train_predicted_humans[idx]["bbox"] = bb
        train_predicted_humans[idx]["category_id"] = max_cat_id + 1

    for i in train_predicted_humans:
        modanet["annotations"].append(i)

    for i in val_predicted_humans:
        modanet["annotations"].append(i)

    image_id_to_image_name = {}
    
    for key in modanet["images"]:
        image_id_to_image_name[int(key["id"])] = key["file_name"]
        
    remove_categories = ["sunglasses", "belt", "scarf/tie"]

    remove_category_ids = list()

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

    modanet_train, modanet_val = {
                                     "images": [],
                                     "annotations": []
                                 }, {
                                     "images": [],
                                     "annotations": []
                                 }

    modanet_train["images"] = []
    modanet_val["images"] = []

    modanet_imgs = modanet["images"][:int(percentage * len(modanet["images"]))]

    random.shuffle(modanet_imgs)

    val_ratio = 0.1

    modanet_train["images"] = modanet_imgs[:int((1 - val_ratio) * len(modanet_imgs))]
    modanet_val["images"] = modanet_imgs[int((1 - val_ratio) * len(modanet_imgs)):]

    print("modanet train / val:", len(modanet_train["images"]), len(modanet_val["images"]))

    highest_category_id = max([i["id"] for i in modanet["categories"]])

    # Adjust the ids of the second dataset
    modanet_train["annotations"] = annotations_from_images(modanet["annotations"], modanet_train["images"], "modanet_train")
    modanet_val["annotations"] = annotations_from_images(modanet["annotations"], modanet_val["images"], "modanet_val")

    train = {
        "images": modanet_train["images"],
        "annotations": modanet_train["annotations"],
        "categories": modanet["categories"]
    }

    val = {
        "images": modanet_val["images"],
        "annotations": modanet_val["annotations"],
        "categories": modanet["categories"]
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
        shutil.copy(os.path.join("data", "modanet", "images", filename),
                        os.path.join("data", "images", "val", filename))

    for img in train["images"]:
        filename = img["file_name"]
        shutil.copy(os.path.join("data", "modanet", "images", filename),
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
        val_image_annotations[image_id_to_image_filename[int(ann["image_id"])]] = []

    for ann in val["annotations"]:
        t = list()
        t.append(ann["category_id"])
        t.extend(ann["bbox"])
        val_image_annotations[image_id_to_image_filename[ann["image_id"]]].append(t)

    for ann in train["annotations"]:
        train_image_annotations[image_id_to_image_filename[ann["image_id"]]] = []

    for ann in train["annotations"]:
        t = list()
        t.append(ann["category_id"])
        t.extend(ann["bbox"])
        train_image_annotations[image_id_to_image_filename[ann["image_id"]]].append(t)

    for idx, bboxes in val_image_annotations.items():
        if type(idx) != str and idx < 1_000_000:
            idx = "0" + str(idx)
        im = cv2.imread(f"data/images/val/{idx}")
        with open(f"data/labels/val/{idx.split('.')[0]}.txt", "w") as f:
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
        im = cv2.imread(f"data/images/train/{idx}")
        with open(f"data/labels/train/{idx.split('.')[0]}.txt", "w") as f:
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
