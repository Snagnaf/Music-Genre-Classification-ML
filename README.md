# Music Genre Classification ML
Implementation of the paper [Music Genre Classification Using Machine Learning Techniques](https://arxiv.org/abs/2509.01762).
Project for the **Machine Learning** exam (6 CFU) of the Master's Degree in **Telecommunication Engineering** - **University of Rome "La Sapienza"**.

## Project Overview
This project implements a system for classifying music genres using Machine Learning and Deep Learning techniques. It extracts audio features (such as MFCC, Chroma, Spectral Centroid) from the **GTZAN** dataset and trains three different models to compare their performance:

1.  **SVM (Support Vector Machine)**: A geometric classifier using the RBF kernel.
2.  **RF (Random Forest)**: An ensemble of decision trees.
3.  **CNN (Convolutional Neural Network)**: A deep learning model that works on audio spectrograms.

The system includes a graphical **Dashboard** to visualize results (PCA, Confusion Matrices, Feature Importance) and a script to **test** the models on new audio files.

## Features Extraction
For each audio segment, the system extracts a vector of **52 features** that describe the sound DNA:
- **MFCC (26 features)**: 13 Mel-Frequency Cepstral Coefficients (Mean) + 13 (Standard Deviation). They capture the "timbre" of the sound.
- **Chroma (12 features)**: 12 energy levels for each of the 12 musical notes. Crucial for harmony-based genres (Jazz, Classical).
- **Spectral Contrast (7 features)**: Captures the "brightness" and dynamic range of different frequency bands.
- **Temporal & Spectral Statistics (7 features)**: Zero Crossing Rate, RMS, BPM, Amplitude Variance, Spectral Centroid, Bandwidth, and Rolloff.

## Requirements
Install all dependencies using the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Dataset Setup
This project is designed to work with the **GTZAN Genre Collection** dataset.

1.  **Download** the GTZAN dataset (e.g., from [Kaggle](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)).
2.  **Extract** the `genres_original` folder into the `src/data/` directory.

Your directory structure should look like this:

```
src/data/
└── genres_original/
    ├── blues/
    ├── classical/
    ├── country/
    ├── disco/
    ├── hiphop/
    ├── jazz/
    ├── metal/
    ├── pop/
    ├── reggae/
    └── rock/
```

### 1. Training & Analysis
To extract features, train the models, and generate the performance dashboard, run the main script:

```bash
python src/main.py
```

**What happens:**
*   Audio features are extracted from the dataset (saved to `src/cache/` for faster subsequent runs).
*   Models (SVM, RF, CNN) are trained on the data.
*   Results are evaluated.
*   A dashboard is displayed and saved to `src/images/dashboard.png`.

![loading](https://github.com/user-attachments/assets/208f4191-75e8-4833-b196-bc5274440359)

### 2. Testing on New Songs
To classify your own music files:

1.  Place your audio files (`.wav`, `.mp3`) in the `src/songs/` folder.
2.  Run the test script:

```bash
python src/test.py
```

The script will output the predicted genre for each song using all three trained models.

![test](https://github.com/user-attachments/assets/24870a3d-4b36-477b-8e61-0bb29e230777)

## Project Structure
```
Music-Genre-Classification-ML/
├── src/
│   ├── application/        # Core logic modules
│   │   ├── models/         # Model definitions (SVM, RF, CNN)
│   │   ├── dashboard.py    # Visualization logic
│   │   ├── features.py     # Feature extraction logic
│   │   └── training.py     # Training routines
│   ├── cache/              # Cached features and trained models
│   ├── data/               # Dataset directory
│   ├── images/             # Generated dashboards
│   ├── songs/              # Folder for user's test songs
│   ├── main.py             # Main entry point (Training)
│   ├── test.py             # Testing script for new songs
│   └── variables.py        # Constants and configuration
└── README.md
```
