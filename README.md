# Clothing Detection
Code for detecting different clothes on a person, and locate them with bounding boxes

## Create environment

> python3 -m venv env && source env/bin/activate

And install dependencies for downloading MIAP

> pip install boto3 botocore tqdm

## Downloading ModaNet

Download only the relevant data - Approx. 2GB of image data:

> ./download_modanet.sh . True

Which contains dataset of the following labels:


Labels
Each polygon (bounding box, segmentation mask) annotation is assigned to one of the following labels:

| Label | Description | Fine-Grained-categories |
| :---: | :---: | :---: |
| 1 | bag | bag |
| 2 | belt | belt |
| 3 | boots | boots |
| 4 | footwear | footwear |
| 5 | outer | coat/jacket/suit/blazers/cardigan/sweater/Jumpsuits/Rompers/vest |
| 6 | dress | dress/t-shirt dress |
| 7 | sunglasses | sunglasses |
| 8 | pants | pants/jeans/leggings |
| 9 | top | top/blouse/t-shirt/shirt |
| 10 | shorts | shorts |
| 11 | skirt | skirt |
| 12 | headwear | headwear |
| 13 | scarf & tie | scartf & tie |

The annotation data format of ModaNet follows the same style as COCO-dataset.

## Downloading MIAP (More Inclusive Annotations for People)

Inside the data folder create a miap folder and change directories to it, we only download the validation dataset because it is enough images to train on for our task (7410 images), while the training dataset consists of over 70 000 images.

> mkdir miap && cd miap

Then download the following files:

> wget https://storage.googleapis.com/openimages/open_images_extended_miap/open_images_extended_miap_boxes_train.csv && wget https://storage.googleapis.com/openimages/open_images_extended_miap/open_images_extended_miap_images_train.lst

Then run the following script

> python downloader.py $IMAGE_LIST_FILE --download_folder=$DOWNLOAD_FOLDER --num_processes=5

Where $IMAGE_LIST_FILE is the .lst file from the above command and the $DOWNLOAD_FOLDER is the folder that we want to download the files to

> mkdir images

Example

> python downloader.py open_images_extended_miap_images_train.lst --download_folder=./images --num_processes=5

## Requirements

- pandas
- opencv-python
- torch
- matplotlib
- albumentations
- torchvision

## Info

For the task we will be using the following classes:

1. Bag
2. Boots
3. Footwear
4. Outer
5. Dress
6. Pants
7. Top
8. Shorts
9. Skirt
10. Headwear
11. Person

To ensure that the clothes detected are on the person, I've implemented a heuristic that detects whether the bounding boxes of the clothes are within the bounding box of the person with a certain threshold.

# Training with COCO

Normal training schedule from _scratch_

> python train.py --workers 8 --device 0 --batch-size 32 --data coco_data/coco.yaml --img 640 640 --cfg cfg/training/yolov7.yaml --weights '' --name yolov7 --hyp coco_data/hyp.scratch.p5.yaml

## Transfer learning

Transfer training schedule from _weights_

> python train.py --workers 8 --device 0 --batch-size 32 --data coco_data/miap.yaml --img 640 640 --cfg cfg/training/yolov7-custom.yaml --weights 'yolov7_training.pt' --name yolov7-custom --hyp coco_data/hyp.scratch.custom.yaml
