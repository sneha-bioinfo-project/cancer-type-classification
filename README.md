# 🧬 Cancer Type Classification from RNA‑Seq

Predicting cancer type from high‑dimensional RNA‑Seq gene expression using classical machine learning on the **UCI Gene Expression Cancer RNA‑Seq** dataset.

---

## 📊 Project Highlights

| Metric                | Value                                   |
|-----------------------|-----------------------------------------|
| Samples Analyzed      | 800+ tumor samples                      |
| Cancer Types          | BRCA, COAD, KIRC, LUAD, PRAD            |
| Genes Considered      | >20,000 expression features (RNA‑Seq)   |
| Features Selected     | Top 1,000 genes (ANOVA F‑test)          |
| Models Trained        | 5 classical ML algorithms               |
| Best Models           | Logistic Regression, SVM, XGBoost       |
| Test Accuracy         | ~99.4% (20% hold‑out)                   |
| 5‑Fold CV Accuracy    | ~99.8% (best models)                    |
| Production Ready      | ✅ Reproducible notebook pipeline       |

---

## 🎯 What This Project Shows

### Machine Learning Skills

- ✅ End‑to‑end supervised classification pipeline (data → model → evaluation).
- ✅ Multi‑algorithm comparison: Logistic Regression, SVM (linear/RBF), RandomForest, XGBoost.
- ✅ High‑dimensional data handling with univariate feature selection (SelectKBest, ANOVA F‑test).
- ✅ Robust evaluation: stratified train/test split + 5‑fold cross‑validation, confusion matrices, class‑wise metrics.

### Bioinformatics / Omics Knowledge

- ✅ Work with RNA‑Seq **gene expression matrices** and cancer subtype labels.
- ✅ Map samples to tumor types (BRCA, COAD, KIRC, LUAD, PRAD) and encode labels for ML.
- ✅ Identify **top predictive genes** via tree‑based feature importance (RandomForest, XGBoost) as candidate biomarkers.
- ✅ Understand that a small subset of genes can separate tumor types with very high accuracy in this dataset.

### Reproducible ML Pipeline

- ✅ Clean project structure: `notebooks/`, `data/`, `results/`, `models/`.
- ✅ Single Colab notebook that runs from raw CSVs to final results.
- ✅ Result artifacts: `model_performance.tsv`, `top20_genes_rf.tsv`, `top20_genes_xgboost.tsv` for downstream analysis.

---

## 🧪 The Science Behind It

### Why This Dataset?

The UCI Gene Expression Cancer RNA‑Seq dataset is a widely used benchmark for:

- Comparing algorithms on **multi‑class cancer subtype classification**.
- Testing methods for dimensionality reduction and feature selection in omics data.
- Exploring the relationship between gene expression profiles and tumor type.

### Key Modeling Ideas Learned

- A combination of **feature selection** (ANOVA F‑test) and linear / tree‑based models is enough to almost perfectly separate tumor types.
- Ensemble methods (RandomForest, XGBoost) provide **feature importance** scores that highlight a small set of genes driving decisions, which can be linked to cancer biology in further work.
- Cross‑validation is essential to confirm that very high test accuracy (~99%+) is due to true signal and not overfitting.

---

## 🚀 Quick Start

### Option 1 – Run in Google Colab

1. Download `data.csv` and `labels.csv` from the UCI Gene Expression Cancer RNA‑Seq dataset and place them in `data/`.
2. Open `notebooks/Project2_cancer_classification.ipynb` in Google Colab.
3. Update `PROJECT_DIR` to your Drive path if needed.
4. Run all cells:
   - Mount Drive.
   - Load & preprocess data.
   - Train all models.
   - Generate metrics and top‑gene tables.

### Option 2 – Local Jupyter


Clone repository
git clone https://github.com/sneha-bioinfo-project/cancer-type-classification-rnaseq.git
cd cancer-type-classification-rnaseq

(Optional) create and activate a virtual environment, then install deps
pip install -r requirements.txt # if you add this file

Launch Jupyter
jupyter notebook notebooks/Project2_cancer_classification.ipynb

text

Place `data.csv` and `labels.csv` in `data/` before running.

---

## 📊 Model Performance

### Algorithm Comparison (20% Test Split)

| Algorithm           | Accuracy | Notes                      |
|---------------------|----------|----------------------------|
| Logistic Regression | ~99.4%   | Strong linear baseline     |
| SVM (linear)        | ~99.4%   | Margin‑based classifier    |
| SVM (RBF)           | ~99.4%   | Captures non‑linearities   |
| Random Forest       | ~98.8%   | Robust, slightly lower     |
| XGBoost             | ~99.4%   | Powerful gradient boosting |

All models also show ~99.6–99.8% average accuracy in 5‑fold cross‑validation, confirming stable performance.

---

## 💻 Technologies Used

- Python 3
- Pandas, NumPy
- Scikit‑learn (classification, feature selection, metrics)
- XGBoost (multi‑class gradient boosting)
- Matplotlib, Seaborn
- Jupyter / Google Colab

---

## 📚 References

- UCI Gene Expression Cancer RNA‑Seq dataset.
- Scikit‑learn documentation.
- XGBoost documentation.
- Literature on feature importance and biomarker discovery from gene expression data.
## 📁 Project Structure

