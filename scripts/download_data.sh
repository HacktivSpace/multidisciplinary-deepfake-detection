#!/bin/bash

set -euo pipefail

DATA_DIR="data/raw"
PROCESSED_DIR="data/processed"
LOG_DIR="logs"

# Download URLs
IMAGE_DOWNLOAD_URL="https://www.kaggle.com/code/mytechnotalent/deepfake-detector/input"
AUDIO_DOWNLOAD_URL="https://www.kaggle.com/code/iasadpanwhar/deep-fake-audio-classification-hybrid-architecture/input"
DEEPFAKE_VIDEO_URLS=(
    "https://osf.io/6rvjf"
    "https://osf.io/43m2z"
    "https://osf.io/jdpkq"
    "https://osf.io/urq2m"
    "https://osf.io/pc7rz"
    "https://osf.io/dwzjy"
    "https://osf.io/hmv7x"
    "https://osf.io/rk5tm"
    "https://osf.io/u9bps"
    "https://osf.io/adu7y"
)
REAL_VIDEO_URLS=(
    "https://osf.io/ae5u2"
    "https://osf.io/n7rzp"
    "https://osf.io/38bre"
    "https://osf.io/ptvus"
    "https://osf.io/xa475"
    "https://osf.io/judfp"
    "https://osf.io/xa39z"
    "https://osf.io/84pkv"
    "https://osf.io/7uwjg"
    "https://osf.io/bpjyt"
)

# To create directories if they don't exist
mkdir -p $DATA_DIR/images
mkdir -p $DATA_DIR/audios
mkdir -p $DATA_DIR/videos/deepfake
mkdir -p $DATA_DIR/videos/real
mkdir -p $PROCESSED_DIR
mkdir -p $LOG_DIR

LOG_FILE="$LOG_DIR/download_data.log"

exec > >(tee -i $LOG_FILE)
exec 2>&1

echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] Starting data download."

echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] Downloading image dataset from $IMAGE_DOWNLOAD_URL."
curl -o "$DATA_DIR/images/dataset.zip" -L $IMAGE_DOWNLOAD_URL

if [ $? -ne 0 ]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') [ERROR] Failed to download image dataset from $IMAGE_DOWNLOAD_URL."
    exit 1
fi

echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] Unzipping image dataset."
unzip -o "$DATA_DIR/images/dataset.zip" -d $DATA_DIR/images

if [ $? -ne 0 ]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') [ERROR] Failed to unzip image dataset."
    exit 1
fi

rm "$DATA_DIR/images/dataset.zip"

echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] Downloading audio dataset from $AUDIO_DOWNLOAD_URL."
curl -o "$DATA_DIR/audios/dataset.zip" -L $AUDIO_DOWNLOAD_URL

if [ $? -ne 0 ]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') [ERROR] Failed to download audio dataset from $AUDIO_DOWNLOAD_URL."
    exit 1
fi

echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] Unzipping audio dataset."
unzip -o "$DATA_DIR/audios/dataset.zip" -d $DATA_DIR/audios

if [ $? -ne 0 ]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') [ERROR] Failed to unzip audio dataset."
    exit 1
fi

rm "$DATA_DIR/audios/dataset.zip"

for url in "${DEEPFAKE_VIDEO_URLS[@]}"; do
    echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] Downloading deepfake video from $url."
    curl -o "$DATA_DIR/videos/deepfake/$(basename $url)" -L $url
    
    if [ $? -ne 0 ]; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') [ERROR] Failed to download deepfake video from $url."
        exit 1
    fi
done

for url in "${REAL_VIDEO_URLS[@]}"; do
    echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] Downloading real video from $url."
    curl -o "$DATA_DIR/videos/real/$(basename $url)" -L $url
    
    if [ $? -ne 0 ]; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') [ERROR] Failed to download real video from $url."
        exit 1
    fi
done

echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] Data download and extraction process completed successfully."
