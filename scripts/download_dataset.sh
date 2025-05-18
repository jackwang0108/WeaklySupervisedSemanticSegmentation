#!/bin/bash

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)
DATASET_DIR="${ROOT_DIR}/data"

# PASCAL VOC 2012
echo "Downloading PASCAL VOC 2012 dataset..."
VOC_DIR="${DATASET_DIR}/VOC2012"
mkdir -p "${VOC_DIR}"
wget -c -q --show-progress --no-check-certificate \
    -O "${VOC_DIR}/VOCtrainval_11-May-2012.tar" \
    "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"

echo "Extracting PASCAL VOC 2012 dataset..."
tar -xf "${VOC_DIR}/VOCtrainval_11-May-2012.tar" -C "${VOC_DIR}"

echo "Downloading PASCAL VOC 2012 augmented annotations by SBD dataset"
wget -c -q --show-progress --no-check-certificate \
    -O "${VOC_DIR}/SegmentationClassAug.zip" \
    "https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip?dl=1"

echo "Extracting augmented annotations..."
unzip -q "${VOC_DIR}/SegmentationClassAug.zip" -d "${VOC_DIR}/VOCdevkit/VOC2012/"

echo "${VOC_DIR}/VOCdevkit/VOC2012" >"${ROOT_DIR}/datasets/voc/voc.txt"

# MS COCO 2014
echo "Downloading MS COCO 2014 dataset..."
COCO_DIR="${DATASET_DIR}/COCO2014"
mkdir -p "${COCO_DIR}"
wget -c -q --show-progress --no-check-certificate \
    -O "${COCO_DIR}/train2014.zip" \
    "http://images.cocodataset.org/zips/train2014.zip"
wget -c -q --show-progress --no-check-certificate \
    -O "${COCO_DIR}/val2014.zip" \
    "http://images.cocodataset.org/zips/val2014.zip"
wget -c -q --show-progress --no-check-certificate \
    -O "${COCO_DIR}/annotations_trainval2014.zip" \
    "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"

echo "Extracting MS COCO 2014 dataset..."
unzip -q "${COCO_DIR}/train2014.zip" -d "${COCO_DIR}"
unzip -q "${COCO_DIR}/val2014.zip" -d "${COCO_DIR}"
unzip -q "${COCO_DIR}/annotations_trainval2014.zip" -d "${COCO_DIR}"

echo "${COCO_DIR}" >"${ROOT_DIR}/datasets/coco/coco.txt"
