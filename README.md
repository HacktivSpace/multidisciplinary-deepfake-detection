# Multidisciplinary Deepfake Detection

This repository contains a solution for detecting deepfakes across multiple modalities, including images, audio, and video. The system leverages various machine learning models, including CNNs, Transformers, SVMs, Bayesian models, and Vision Transformers, to classify real and fake data effectively.

## Table of Contents

- [Project Overview](#project-overview)
- [Directory Structure](#directory-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Project](#running-the-project)
- [Models](#models)
- [Notebooks](#notebooks)
- [Logging](#logging)
- [Docker Support](#docker-support)
- [License](#license)

## Project Overview

This project is designed to detect deepfakes using a combination of different models applied to image, audio, and video data. It includes:
- **Image Classification** using CNNs and Vision Transformers.
- **Audio Classification** using advanced models and preprocessing techniques.
- **Video Classification** by analyzing frames using deep learning models.
- **NLP for Text Analysis** in videos where necessary.

## Directory Structure

The repository is organized as follows:

```
multidisciplinary-deepfake-detection/
│
├── data/
│   ├── raw/                # Raw data
│   ├── processed/          # Processed data
│   └── sample_data.csv     # Example data file
│
├── models/
│   ├── saved_models/       # Trained models
│   ├── cnn_model.h5        # CNN model
│   ├── transformer_model.pth # Transformer model
│   ├── svm_model.pkl       # SVM model
│   ├── bayesian_model.pkl  # Bayesian model
│   ├── vision_transformer_model.pth # Vision Transformer model
│   └── model_architecture.png # Model architecture visualization
│
├── notebooks/              # Jupyter notebooks for EDA, training, and evaluation
│   ├── Data Preprocessing.ipynb
│   ├── Exploratory Data Analysis.ipynb
│   ├── Model Training.ipynb
│   └── Model Evaluation.ipynb
│
├── scripts/                # Shell and Python scripts
│   ├── download_data.sh
│   ├── preprocess_data.py
│   ├── generate_report.py
│   ├── train_all_models.sh
│   └── evaluate_all_models.sh
│
├── src/                    # Source code for models, data processing, and utilities
│   ├── dataset/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   ├── utils/
│   ├── processing/
│   └── config.py
│
├── tests/                  # Unit tests for the project
│   ├── test_data_loading.py
│   ├── test_model.py
│   ├── test_training.py
│   ├── test_evaluation.py
│   └── test_utils.py
│
├── logs/                   # Log files for tracking the progress
│   ├── model_training.log
│   ├── data_preprocessing.log
│   ├── evaluation.log
│   └── system.log
│
├── Dockerfile              # Docker configuration
├── docker-compose.yml      # Docker Compose configuration
├── requirements.txt        # Python dependencies
├── setup.py                # Python package setup
├── .env                    # Environment variables
├── entrypoint.sh           # Docker entrypoint script
├── LICENSE                 # License file
├── .gitattributes          # Git attributes
├── .gitignore              # Git ignore rules
├── CHANGELOG.md            # Changelog for the project
└── README.md               # Project documentation
```

## Getting Started

### Prerequisites

- **Python 3.9** or higher.
- **Docker** and **Docker Compose** installed.
- **Git** for version control.

### Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/HacktivSpace/multidisciplinary-deepfake-detection.git
    cd multidisciplinary-deepfake-detection
    ```

2. **Set up the environment:**

    Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

3. **Download the dataset:**

    Run the provided script to download the necessary datasets:
    ```bash
    bash scripts/download_data.sh
    ```

### Running the Project

#### Locally

To train the models locally:

1. **Preprocess the data:**
    ```bash
    python scripts/preprocess_data.py
    ```

2. **Train the models:**
    ```bash
    bash scripts/train_all_models.sh
    ```

3. **Evaluate the models:**
    ```bash
    bash scripts/evaluate_all_models.sh
    ```

#### With Docker

Alternatively, one can run the entire setup using Docker:

1. **Build the Docker image:**
    ```bash
    docker-compose build
    ```

2. **Run the Docker container:**
    ```bash
    docker-compose up
    ```

## Models

The project includes several machine learning models:

- **CNN Model** for image classification.
- **Transformer Model** for handling sequential data.
- **SVM Model** for baseline classification tasks.
- **Bayesian Model** for probabilistic modeling.
- **Vision Transformer Model** for advanced image classification tasks.

## Notebooks

The following Jupyter notebooks are provided for further exploration:

- **Data Preprocessing:** Contains steps for cleaning and preparing the data.
- **Exploratory Data Analysis:** Includes visualizations and insights from the dataset.
- **Model Training:** Contains code for training the models.
- **Model Evaluation:** Shows the evaluation results of the trained models.

## Logging

Logs for all major processes are stored in the `logs/` directory. This includes logs for:

- Data Preprocessing
- Model Training
- Model Evaluation
- System Setup and Execution

## Docker Support

This project supports Docker to simplify setup and deployment. The `Dockerfile` and `docker-compose.yml` are configured to run the application in a containerized environment.

- The `Dockerfile` handles environment setup and installation of dependencies.
- The `docker-compose.yml` file orchestrates the various services, such as the web app and database.

## License

This project is licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International Public License. By using this software, you agree to the terms stated in the [LICENSE](LICENSE) file.
```
