#!/bin/bash

# To setup the environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate deepfake-detection

LOG_DIR="logs"
TRAIN_LOG="$LOG_DIR/train_all_models.log"

mkdir -p $LOG_DIR

exec > >(tee -i $TRAIN_LOG)
exec 2>&1

echo "===================================="
echo "Starting training of all models"
echo "Date: $(date)"
echo "===================================="
echo ""

echo "Training CNN model..."
python -c "
from src.train import train_cnn
train_cnn()
"
if [ $? -eq 0 ]; then
    echo "CNN model training completed successfully."
else
    echo "CNN model training failed."
    exit 1
fi
echo ""

echo "Training Transformer model..."
python -c "
from src.train import train_transformer
train_transformer()
"
if [ $? -eq 0 ]; then
    echo "Transformer model training completed successfully."
else
    echo "Transformer model training failed."
    exit 1
fi
echo ""

echo "Training SVM model..."
python -c "
from src.train import train_svm_model
train_svm_model()
"
if [ $? -eq 0 ]; then
    echo "SVM model training completed successfully."
else
    echo "SVM model training failed."
    exit 1
fi
echo ""

echo "Training Bayesian model..."
python -c "
from src.train import train_bayesian
train_bayesian()
"
if [ $? -eq 0 ]; then
    echo "Bayesian model training completed successfully."
else
    echo "Bayesian model training failed."
    exit 1
fi
echo ""

echo "Training Vision Transformer model..."
python -c "
from src.train import train_vision_transformer
train_vision_transformer()
"
if [ $? -eq 0 ]; then
    echo "Vision Transformer model training completed successfully."
else
    echo "Vision Transformer model training failed."
    exit 1
fi
echo ""

echo "===================================="
echo "Training of all models completed"
echo "Date: $(date)"
echo "===================================="
