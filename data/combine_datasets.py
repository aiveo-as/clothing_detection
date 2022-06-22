import os
import json
import random
import shutil

from typing import List
from tqdm import tqdm

random.seed(1)

def annotations_from_images(annotations, images, name) -> List:

    new_annotations = []

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
    miap = json.load(open("data/miap/instances_all_miap.json", "r"))
    modanet = json.load(open("data/modanet/instances_all_modanet_transformed.json", "r"))

    # We do not want all categories in modanet, therefore we remove the following categories

    remove_categories = ["sunglasses", "belt", "scarf/tie"]

    remove_category_ids = []

    for idx, category in enumerate(modanet["categories"]):
        if category["name"] in remove_categories:
            remove_category_ids.append(category["id"])
            modanet["categories"].pop(idx)

    for idx, annotation in enumerate(modanet["annotations"]):
        if annotation["category_id"] in remove_category_ids:
            modanet["annotations"].pop(idx)

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

    percentage = 0.01

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

    modanet_train["annotations"] = annotations_from_images(modanet["annotations"], modanet_train["images"], "modanet_train")
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

    json.dump(train, open("data/combined_train.json", "w"))
    json.dump(val, open("data/combined_val.json", "w"))

    if os.path.exists("data/val"):
        shutil.rmtree("data/val")
    
    os.mkdir("data/val")

    for img in val["images"]:
        filename = img["file_name"]
        try:
            shutil.copy(os.path.join("data", "modanet", "images", filename), os.path.join("data", "val", filename))
        except:
            shutil.copy(os.path.join("data", "miap", "images", filename), os.path.join("data", "val", filename))

    if os.path.exists("data/train"):
        shutil.rmtree("data/train")
    
    os.mkdir("data/train")

    for img in train["images"]:
        filename = img["file_name"]
        try:
            shutil.copy(os.path.join("data", "modanet", "images", filename), os.path.join("data", "train", filename))
        except:
            shutil.copy(os.path.join("data", "miap", "images", filename), os.path.join("data", "train", filename))
        
    
