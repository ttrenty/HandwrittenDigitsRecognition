# Handwritten Digits Recognition

This project uses a K-Nearest Neighbors (KNN) model to perform handwritten digits recognition. It allows users to use a pretrained model or adapt it to their writing by providing images.

Disclaimer: The following work represents an educational endeavor. And an early one of mines, do not expect sky-high performances from it.

## Project Description

The project is structured as follows:

- `digit_recognition.py`: The main script to run the digit recognition.
- `lib/`: This directory contains the necessary libraries for the project.
  - `image_processing.py`: Contains functions for processing images.
  - `knn.py`: Contains the implementation of the K-Nearest Neighbors algorithm.
- `data/`: This directory contains the testing and training data.
  - `testing/`: Place your images here for testing.
  - `training/`: Contains the training data. You can add your own images to the `user/` directory to adapt the model to your handwriting.
- `img/`: This directory contains example images and features. The `KNN_euclidean_distance_matrix.png` file represents the similarities between the digits of the training set for the considered features.

## Installation

1. Clone this repository.
2. Navigate to the project directory.
3. Install the necessary Python packages using pip:

```bash
python3 -m venv project
source project/bin/activate
pip install -r requirements.txt
```

## Usage
Run the digit_recognition.py script and follow the prompts in English.

```bash
python3 digit_recognition.py
```

## Troubleshooting
Images given should not be too high quality as the application automatically lowers it and it could make the individual digits unrecognizable.

## Changelog:
Project written in 2021-04. Adapted in English in 2024-01.