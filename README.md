# Data Engineer Assignment: Language & Dialect Dataset Engineering for Speech Classification Model Fine-Tuning

## Data

This project prepares a multilingual and multi-dialect audio dataset ([Common Voice](https://commonvoice.mozilla.org/en/datasets) - publicly available voice dataset, powered by the voices of volunteer contributors around the world. People who want to build voice applications can use the dataset to train machine learning models) for machine learning experiments.


We selected the following **languages and dialects** based on availability and language diversity (to have combination of both languages and dialects):
 

In this case we used just data from validated.tsv


### Dialect Coverage (Before Balancing)

| Language   | Dialect                                                | Clips | Duration (sec) |
|------------|---------------------------------------------------------|--------|----------------|
| English    | England English                                         | 36     | 222.4          |
| English    | Southern African (South Africa, Zimbabwe, Namibia)     | 9      | 44.4           |
| English    | Australian English                                      | 9      | 51.9           |
| French     | Français de France                                      | 77     | 295.6          |
| French     | Français de France, Français de Belgique                | 24     | 124.2          |
| French     | Français de France, Accent du Sud Ouest, accent plat   | 20     | 92.2           |
| German     | Deutschland Deutsch                                     | 94     | 297.9          |
| German     | Österreichisches Deutsch                                | 57     | 296.4          |
| German     | Schweizerdeutsch                                        | 14     | 82.7           |
| Slovenian  | (no dialect label)                                      | 7      | 33.2           |


---

## Dialect Coverage

We ensured:
- 3 major dialects per major language (where available)
- up to 5 minutes of validated audio per dialect

---

## Limitations

- Some dialects have very limited audio data (e.g. up to 44 sec)
- sentence domain, gender and variant and segment - often missing
- Recording quality varies (e.g., background noise) - from few examples manually tested

---

## How your pipeline could scale to support 100+ dialects, Options:

### Rewrriting function
Batch Processing: Current functiondo not support batch mode,  we proces files one by one
Logging and error handling: implement logging and recovery logic


### Downloading and Storing Massive Datasets
Automatization of dowloads: For now we did it manually
Storage: Saving data to cloud storages for scalable access


### Processing Power
GPU - extractions, coverting, models
PySpark - loading, filtering etc

---


### Overview

| File/Folder               | Description                                                                                                                                |
|---------------------------|--------------------------------------------------------------------------------------------------------------------------------------------|
| `data/`                   | Raw Common Voice dataset files organized by language (e.g., `validated.tsv`, `clip_durations.tsv`, and MP3 audio clips).                   |
| `exports/`                | Contains exported versions of the final dataset manifest in multiple formats (CSV, JSON, Hugging Face Dataset).                            |
| `plots/`                  | Generated waveform and spectrogram images for visual inspection of audio sample quality and diversity.                                     |
| `processed_audio/`        | Converted audio files in mono WAV, 16 kHz format, organized by language and dialect for model input.                                       |
| `chatGPT-LLM_usage/`      | Notes describing which parts of the pipeline were supported by ChatGPT/LLMs, and how outputs were validated.                               |
| `final_manifest.csv`      | The final cleaned dataset manifest including audio paths, language, dialect, speaker ID, and duration.                                     |
| `load_data.ipynb`         | Jupyter Notebook used for loading, preprocessing, testing, and inspecting dataset structure, converting to .wav format and visualizations. |
| `multi_format_export.py`  | Script to export the dataset manifest into various formats like CSV, JSON, or Hugging Face format.                                         |
| `optional_enhacements.py` | Augmentation and dialect balancing functions (e.g., pitch shift, speed change, background noise injection).                                |
| `spectogram_dashboard.py` | Streamlit-based dashboard to explore audio samples, metadata, and spectrograms by language/dialect.                                        |


---

## Setup Instructions

Follow the steps below to set up and run the project:

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Create and Activate a Virtual Environment (Optional but Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate it
# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

```

### 3.Install Dependencies
Please install the following packages before running the code:

- `pandas`
- `numpy`
- `matplotlib`
- `streamlit`
- `torchaudio`
- `pydub`
- `tqdm`
- `soundfile`
- `librosa`

### 4.Running .ipynb Notebooks
Open the file and run the cells interactively

### 5. Running multi_format_export.py
To execute a Python script from the command line or terminal, use:
```bash
python multi_format_export.py
```

### 6. Run the Streamlit Dashboard
```bash
streamlit run spectogram_dashboard.py
```




---

# Author
### Tomislav Seljan
