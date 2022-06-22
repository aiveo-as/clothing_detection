import csv
import os
import json

import pandas as pd
import cv2 as cv
from tqdm import tqdm
import numpy as np

if __name__ == "__main__":

    csv_path = "data/miap/open_images_extended_miap_boxes_val.csv"

    df = pd.read_csv(csv_path)

    print(f"Found a total of {len(df)} of annotations for {len(df['ImageID'].unique())} images")
    print("\n###############")
    print("Starting to create COCO annotations")
    print("###############\n")

    coco_annotations = {}

    coco_annotations["info"] = {
        "name": "MIAP (More Inclusive Annotations for People)",
        "url": "https://storage.googleapis.com/openimages/web/extended.html",
        "version": "1.0",
        "year": "2021",
        "contributor": "C. Schumann, S. Ricco, U. Prabhu, V. Ferrari, C. Pantofaru - A Step Toward More Inclusive People Annotations for Fairness",
        "date_created": "2022/06/14"
    }

    coco_annotations["licenses"] = []

    # Adding images

    distinct_image_ids = df["ImageID"].unique()

    coco_annotations["images"] = []

    img_dims = {}

    for _, image_id in tqdm(enumerate(distinct_image_ids), total=len(distinct_image_ids)):
        filename = image_id + ".jpg"

        folderpath = "data/miap/images"

        filepath = os.path.join(folderpath, filename)

        image = cv.imread(filepath)

        img_h, img_w = image.shape[0], image.shape[1]

        img_dims[image_id] = {"img_h": img_h, "img_w": img_w}

        image_info = {
            "file_name": filename,
            "height": img_h,
            "width": img_w,
            "id": image_id
        }

        coco_annotations["images"].append(image_info)

    
    coco_annotations["categories"] = [{"supercategory": "person", "id": 1, "name": "person"}]

    coco_annotations["annotations"] = []

    for i, annotation in tqdm(df.iterrows(), total=len(df)):

        ann_id = annotation["ImageID"]

        img_h, img_w = img_dims[ann_id]["img_h"], img_dims[ann_id]["img_w"]

        x1, x2, y1, y2 = annotation["XMin"] * img_w, annotation["XMax"] * img_w, annotation["YMin"] * img_h, annotation["YMax"] * img_h

        if x2 > img_w:
            x2 = img_w
         
        if y2 > img_h:
            y2 = img_h

        ann_info = {
            "image_id": ann_id,
            "bbox": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
            "category_id": 1
        }

        coco_annotations["annotations"].append(ann_info)

    
    json.dump(coco_annotations, open("data/miap/instances_all_miap.json", "w"))