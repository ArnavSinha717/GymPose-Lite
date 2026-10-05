#!/bin/bash
# Download COCO 2017 Keypoints dataset
# Run from project root: bash data/download_coco.sh

set -e

DEST=~/GymPose-Lite/datasets/coco
mkdir -p "$DEST"
cd "$DEST"

echo "Downloading COCO 2017 annotations..."
wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip -o annotations_trainval2017.zip

echo "Downloading COCO 2017 val images (~1GB)..."
wget -c http://images.cocodataset.org/zips/val2017.zip
unzip -o val2017.zip

echo "Downloading COCO 2017 train images (~18GB)..."
wget -c http://images.cocodataset.org/zips/train2017.zip
unzip -o train2017.zip

echo "Done. Dataset at: $DEST"
echo "  $DEST/train2017/         (~118K images)"
echo "  $DEST/val2017/           (~5K images)"
echo "  $DEST/annotations/       (keypoints JSONs)"
