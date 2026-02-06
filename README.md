# PhysioNet Challenge 2012: ICU Mortality Prediction with LSTM

A PyTorch implementation of a 2-layer Stacked LSTM to predict in-hospital mortality using the **PhysioNet/Computing in Cardiology Challenge 2012** dataset. This project processes multivariate clinical time-series data to classify patient survival.

## 📌 Project Overview

* **Goal:** Predict mortality for ICU patients based on 41 physiological measurements (e.g., Heart Rate, GCS, pH).
* **Model:** Long Short-Term Memory (LSTM) network designed for temporal sequence data.
* **Optimization:** Uses **AdamW** optimizer and **Class Weighted Loss** to handle the severe imbalance between survivors and non-survivors.

## 📂 Repository Structure

```text
├── code/
│   ├── train.py                    # Main training loop (Load data, train model, save checkpoint)
│   ├── test.py                     # Evaluation script for generating predictions
│   ├── data_processor.py           # Script to process set-a (training data)
│   └── data_proccessor_for_test.py # Script to process set-b (test data)
├── data/
│   ├── set-a/                      # Raw training data files
│   ├── set-b/                      # Raw validation/test data files
│   ├── Outcomes-a.txt              # Labels for set-a
│   └── Outcomes-b.txt              # Labels for set-b
├── best_physionet_lstm.pth         # The trained model checkpoint
├── requirements.txt                # Python dependencies
└── .gitignore

```

*> **Note:** The `Processed_Data/` directory is not included in this repository to save space. It will be generated automatically when you run the data processing scripts.*

## 🚀 Installation & Setup

1. **Clone the repository:**
```bash
git clone <your-repo-url>
cd <your-repo-name>

```


2. **Install dependencies:**
It is recommended to use a virtual environment.
```bash
pip install -r requirements.txt

```



## ⚙️ Data Preparation (Required)

Before training, you must convert the raw text files in `data/` into numpy arrays.

1. **Generate Training Data:**
Run the processor to convert `set-a` into training tensors.
```bash
python code/data_processor.py

```


*This will create the `Processed_Data/` folder and save `X_train.npy` and `y_train.npy`.*
2. **Generate Test Data:**
Prepare the validation/test set (`set-b`).
```bash
python code/data_proccessor_for_test.py

```



## 🧠 Training the Model

Once the data is generated, run the training loop:

```bash
python code/train.py

```

* **Outputs:** The script prints training loss and validation AUC per epoch.
* **Artifacts:** The best model weights are saved to `best_physionet_lstm.pth`.

## 📉 Evaluation

To evaluate the model on the test set:

```bash
python code/test.py

```

## 🛠️ Key Technical Features

* **Imbalance Handling:** Uses `BCEWithLogitsLoss` with calculated positive weights.
* **Regularization:** Implements Dropout (0.5) and Weight Decay (via AdamW).
* **Stability:** Includes Gradient Clipping to prevent exploding gradients.

## 📜 License

This project uses data from the PhysioNet 2012 Challenge.

```

```
