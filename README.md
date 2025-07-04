# Fraudulent Website Classification

## 📌 Overview

**Fraudulent Website Classification** is a machine learning-based project focused on detecting and classifying **fraudulent e-commerce websites** from legitimate ones. This project leverages a **crowdsourced dataset** and introduces a novel **archival data extraction pipeline** that enables feature extraction even from dead domains using data from:

- [Wayback Machine](https://archive.org/web/)
- [Past WHOIS APIs](https://whois-history.whoisxmlapi.com/)

Additionally, we introduce a **domain trust scoring mechanism** that provides a user-friendly numeric trust score to aid decision-making about a domain's legitimacy.

---

## 📂 Datasets

We make use of three primary datasets:

- **`primary_dataset.csv`** – The core dataset containing original domain features.
- **`historical_dataset.csv`** – Dataset built by extracting historical snapshots of domains via archival data.
- **`augmented_dataset.csv`** – Combined dataset created by merging the primary and historical datasets.

---

## 🔍 Key Features

- ✅ Classifies domains as **fraudulent** or **legitimate**.
- 🪦 Extracts features from **dead domains** using our custom archival pipeline.
- ⏳ Leverages **temporal domain versions** for dataset augmentation.
- ⭐ Assigns **trust scores** to domains to assist human judgment.
- 🔁 Implements **5-fold cross-validation** for robust evaluation.
- 🧠 Includes **K-Fold Stacking Ensemble Model** with tuned base learners.
- 🧪 All models are **hyperparameter optimized** for improved performance.

---

## 🧪 How to Run

### 🔹 Option 1: Baseline Models

Run the basic classification model using:

```bash
python model.py
```
- This script loads the dataset, trains standard models, and evaluates them. 
- Thoroughly commented and beginner-friendly.


### 🔹 Option 2: Advanced Models with Cross-Validation & Stacking

Run the model with 5 fold cross-validation & Stacking using:

```bash
python model_w_KfoldStacking.py
```
- Implements an ensemble learning approach using stacking with multiple base learners.
- Performs 5-fold cross-validation for model robustness.
- Incorporates hyperparameter tuning for each base model.
- Well-documented through in-line comments for clarity and experimentation.



## 📈 Future Directions

- 🔧 Real-time domain evaluation interface or browser extension
- 🤝 Community-based continual dataset expansion
- 📡 API endpoints for domain classification and trust scoring


## 🤝 Contributing
Contributions are welcome! If you’d like to improve this project:

- Fork the repository
- Create a new branch (git checkout -b feature/my-feature)
- Commit your changes
- Push your branch (git push origin feature/my-feature)
- Open a pull request 🎉

## 👨‍💻 Author
- Kartik Gajaria

