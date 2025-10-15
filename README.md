

The notebook appears to implement a complete **Speech and Emotion Recognition System**.

-----

# Speech and Emotion Recognition System

A comprehensive Python-based project implementing a full-stack speech processing pipeline, including acoustic feature extraction, noise filtering, speech-to-text conversion, and a trained model for emotional detection from audio. This project utilizes the **Toronto Emotional Speech Set (TESS)** dataset.

## Table of Contents

  - [Features](#features)
  - [Dataset](#dataset)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Model & Technologies](#model--technologies)
  
-----

## Features

The system implements the following key functionalities:

  * **Acoustic Analysis:** Functions to extract critical acoustic features from audio, such as Zero Crossing Rate (ZCR), Pitch, Energy (RMS), Spectral Centroid, MFCCs, Chroma Features, and Spectral Contrast.
  * **Speech Recognition:** Integration with the `SpeechRecognition` library (using Google Speech Recognition API) for converting audio segments to text.
  * **Noise Filtering:** Implementation of noise reduction techniques, including pre-emphasis filtering and spectral gating/threshold-based noise gates.
  * **Emotional Detection:** Training and evaluation of an **MLPClassifier** (Multi-layer Perceptron) using `scikit-learn` for classifying emotions from extracted acoustic features.
  * **Real-Time Processing:** A class (`RealTimeAudioProcessor`) demonstrating how to capture and process audio from a microphone in real-time, performing both speech-to-text and emotion detection.

## Dataset

This project is configured to use the **Toronto Emotional Speech Set (TESS)** dataset. The notebook includes a `setup_tess_dataset()` function that attempts to download and extract the Kaggle dataset, which may require you to upload your Kaggle API key (`kaggle.json`).

## Installation

To run this project locally, you will need a Python environment (preferably using Anaconda or a virtual environment).

1.  **Clone the repository:**

    ```bash
    git clone [YOUR_REPOSITORY_URL]
    cd [YOUR_REPOSITORY_NAME]
    ```

2.  **Install system-level dependencies:**
    The notebook requires `portaudio19-dev` for the `pyaudio` library to work, particularly for the real-time functionality.

    ```bash
    # For Linux (Ubuntu/Debian)
    sudo apt install portaudio19-dev
    ```

3.  **Install Python dependencies:**
    The project uses `pip` to install required libraries like `pyaudio`, `SpeechRecognition`, `librosa`, `numpy`, and `scikit-learn`.

    ```bash
    pip install pyaudio SpeechRecognition numpy pandas matplotlib librosa scikit-learn seaborn tqdm soundfile
    ```

## Usage

The entire pipeline is contained within the `recognition (1).ipynb` Jupyter Notebook.

1.  **Run the Notebook:** Open the notebook in your environment (e.g., JupyterLab or VS Code) and execute the cells sequentially.
2.  **Download Dataset:** The `setup_tess_dataset()` function will guide you through downloading the necessary TESS audio files.
3.  **Train Model:** The `train_emotion_model()` function will process the audio, extract features, train the MLP classifier, and save the trained model artifacts: `emotion_model.pkl` and `emotion_scaler.pkl`.
4.  **Real-Time Demo:** The final `RealTimeAudioProcessor` section will attempt to use your microphone for real-time analysis.

## Model & Technologies

| Category | Component | Details |
| :--- | :--- | :--- |
| **Main Model** | MLPClassifier | Multi-layer Perceptron (Neural Network) for emotion classification. |
| **Acoustic Features** | `librosa` | Used for Zero Crossing Rate, Pitch, Energy (RMS), Spectral Centroid, MFCCs, Chroma Features, and Spectral Contrast extraction. |
| **Speech-to-Text** | `SpeechRecognition` | Google Speech Recognition API integration. |
| **Audio I/O** | `pyaudio`, `wave`, `soundfile` | Handling real-time recording and audio file manipulation. |
| **Data/ML** | `numpy`, `pandas`, `sklearn` | Core numerical, data handling, and machine learning utilities. |

## License

This project is licensed under the MIT License
