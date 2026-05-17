# Log Anomaly Transformer 🧠⚙️

A custom deep-learning pipeline utilizing a Transformer Autoencoder to detect catastrophic system failures in highly imbalanced (97% normal) structured event logs. 

Instead of relying on simple keyword matching or regex, this model learns the deep contextual grammar of healthy system execution. When an unpredictable event occurs, the model's cross-entropy loss spikes, triggering an automated anomaly alert.

## 🚀 Key Features

* **Custom Transformer Architecture:** Built from scratch using TensorFlow/Keras with Multi-Head Attention and Layer Normalization (71,000 parameters).
* **Solves the "Dilution Problem":** Replaces standard sequence-average loss with an aggressive **Max-Token Strategy**, ensuring that a single catastrophic word in a 50-word sequence triggers an alarm without being averaged out by normal words.
* **Dynamic Thresholding:** Utilizes 95th-percentile mathematical calibration on held-out normal sequences rather than static, hard-coded tripwires.
* **Production-Ready Inference:** Separated `train.py` and `inference.py` scripts allow for instant, real-time log grading without retraining.

## 📊 The Dilution Problem & Evaluation Strategy

System logs often contain highly repetitive, normal operational data. If a sequence contains 49 normal tokens and 1 fatal token, taking the `mean` of the cross-entropy loss dilutes the error, allowing the crash to slip past the detector.

To maximize **Recall** and **F1-Score**, this pipeline extracts the 3D probability matrix of the actual sequence and isolates the single highest unexpected token error (`max` strategy). 

### Performance Results
* **Precision:** 1.00 (Zero false alarms on anomaly class)
* **Threshold Calibration:** 95th Percentile of Normal Distribution
* **Evaluation Metrics:** ROC-AUC, F1-Score, Precision, Recall

## 🛠️ Tech Stack

* **Frameworks:** TensorFlow 2.x, Keras
* **Data Processing:** NumPy, Pandas
* **Evaluation:** Scikit-Learn
* **Environment:** Python 3.12 

## 📂 Project Structure

* `data_pipeline.py` - Tokenizes, vectorizes, and batches raw CSV event traces.
* `transformer_model.py` - Defines the custom Multi-Head Attention architecture.
* `train.py` - The primary training loop, saving the best `.keras` weights.
* `inference.py` - The production script that loads the model, applies the Max-Token strategy, calculates the percentile threshold, and generates the final classification report.

## ⚙️ Setup & Installation

**1. Clone the repository:**
```bash
git clone https://github.com/Akshatsingh07/Log-Anomaly-Transformer
cd Log-Anomaly-Transformer

**2. Create a virtual environment:**
```Bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

**3. Install dependencies:**
```Bash
pip install tensorflow numpy pandas scikit-learn
(Note: The datasets Event_traces.csv and anomaly_label.csv are required to run this project locally but are excluded from this repository due to GitHub file size limits. Ensure you have the datasets in the root directory before running).

💻 Usage
Training the Model:
To train the autoencoder on healthy logs and save the weights/vocabulary:

```Bash
python train.py
Running Inference:
To evaluate the dataset using the dynamic thresholding engine:

```Bash
python inference.py
This will output the final configuration to anomaly_threshold.txt and anomaly_strategy.txt for use in a live production environment.