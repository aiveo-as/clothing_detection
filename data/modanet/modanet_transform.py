import csv
import os
import json
from turtle import width

from tqdm import tqdm

if __name__ == "__main__":

    modanet = json.load(open("data/modanet/instances_all_modanet.json", "r"))

    for idx, im in enumerate(modanet["images"]):
        im_id = str(im["id"])
        while len(im_id) < 7:
            im_id = "0" + im_id
        modanet["images"][idx]["id"] = im_id

    for idx, im in enumerate(modanet["annotations"]):
        im_id = str(im["image_id"])
        while len(im_id) < 7:
            im_id = "0" + im_id
        modanet["annotations"][idx]["image_id"] = im_id

    heightwidth = {}

    for image in modanet["images"]:
        heightwidth[str(image["id"])] = {"height": image["height"], "width": image["width"]}
    
    for idx, annotation in tqdm(enumerate(modanet["annotations"]), total=len(modanet["annotations"])):

        bb = annotation["bbox"]
        im_id = str(annotation["image_id"])
        img = heightwidth[im_id]
        annotation["bbox"] = [bb[0], bb[1], bb[2], bb[3]]
        if (bb[0] + bb[2]) >= img["width"]:
            modanet["annotations"][idx]["bbox"][2] = img["width"] - bb[0]
        
        if (bb[1] + bb[3]) > img["height"]:
            modanet["annotations"][idx]["bbox"][3] = img["height"] - bb[1]
        
    json.dump(modanet, open("data/modanet/instances_all_modanet_transformed.json", "w"))

