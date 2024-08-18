All notable changes to this product will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this product adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Multidisciplinary Deepfake Detection]

### Added
- Initial setup for the multidisciplinary deepfake detection project.
- Implemented data loading and preprocessing modules.
- Added image, audio, video, and text processing modules.
- Developed CNN, Transformer, SVM, Bayesian, and Vision Transformer models.
- Integrated model training scripts for each model.
- Created evaluation scripts for model performance assessment.
- Implemented utility functions for logging, file handling, and metrics calculation.
- Added unit tests for data loading, model architecture, training, and evaluation.
- Configured Git attributes and ignored unnecessary files in `.gitignore`.

## [0.1.0] - 2024-08-07

### Added
- Initial project structure with necessary directories: `src`, `data`, `models`, `notebooks`, `scripts`, `tests`, `logs`.
- Configured `.gitattributes` for consistent line endings and handling of large files.
- Configured `.gitignore` to exclude unnecessary files and directories.
- Implemented the following modules:
  - `src/audio_processing.py`: Audio processing functions including loading, MFCC extraction, and feature extraction.
  - `src/video_processing.py`: Video processing functions including frame extraction, preprocessing, and feature extraction.
  - `src/image_processing.py`: Image processing functions including loading, preprocessing, and HOG feature extraction.
  - `src/text_processing.py`: Text processing functions including cleaning, tokenizing, removing stopwords, and lemmatizing.
  - `src/blockchain.py`: Blockchain implementation for data integrity.
  - `src/config.py`: Configuration settings for directories, logging, and model hyperparameters.
  - `src/dsp.py`: Digital signal processing functions including STFT, FFT, and filtering.
  - `src/evaluate.py`: Evaluation scripts for CNN, Transformer, SVM, Bayesian, and Vision Transformer models.
  - `src/nlp.py`: NLP processing functions including text cleaning, tokenizing, and lemmatizing using NLTK and Spacy.
  - `src/train.py`: Training scripts for CNN, Transformer, SVM, Bayesian, and Vision Transformer models.
  - `src/utils.py`: Utility functions for file handling, logging, metrics calculation, and data preprocessing.
- Implemented unit tests:
  - `tests/test_data_loading.py`: Tests for data loading functions.
  - `tests/test_model.py`: Tests for model architectures.
  - `tests/test_training.py`: Tests for model training functions.
  - `tests/test_evaluation.py`: Tests for model evaluation functions.
  - `tests/test_utils.py`: Tests for utility functions.
- Added data and logs for testing purposes.

### Changed
- N/A

### Fixed
- N/A

### Removed
- N/A

## [0.1.1] - 2024-08-11

### Added
- Added more comprehensive unit tests to cover edge cases.
- Included additional preprocessing steps for audio and video data.

### Changed
- Improved model training scripts to handle large datasets more efficiently.
- Updated configuration settings to reflect new directory structure.

### Fixed
- Fixed bug in the audio feature extraction function.
- Corrected paths in the data loading scripts.

### Removed
- Deprecated old data processing scripts.

## [0.1.2] - 2024-08-13

### Added
- Integrated blockchain verification for data integrity checks.
- Improved logging functionality for better debugging.

### Changed
- Refactored image processing module for better performance.
- Updated model evaluation scripts to include ROC-AUC score.

### Fixed
- Fixed issue with loading large video files.
- Resolved memory leak in the transformer training script.

### Removed
- Removed redundant helper functions in the utility module.
