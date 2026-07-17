# Steel Energy Predictor — Dimensionality Reduction (PCA) & FastAPI Web Deployment

**Week 3 Project** | AI/ML Engineering

This repository contains the continuation of the UCI Steel Industry Energy Consumption project. In this phase, we optimized our feature space using Principal Component Analysis (PCA) and successfully deployed the pipeline as a live, interactive FastAPI web application:

1. **Part 1 — Dimensionality Reduction via PCA** (`notebooks/week3_pca.ipynb`)
2. **Part 2 — Interactive FastAPI Web Application & Dashboard** (`main.py`)

Together, these represent the production deployment stage of a machine learning workflow: reducing multicollinearity, compressing features for faster inference, and packaging the model with a beautiful UI for end-users.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Part 1 — Principal Component Analysis (PCA)](#part-1--principal-component-analysis-pca)
- [Part 2 — FastAPI Web Deployment & Dashboard](#part-2--fastapi-web-deployment--dashboard)
- [PCA Model Comparison Summary](#pca-model-comparison-summary)
- [Author](#author)

---

## Project Overview

| Feature | Details |
|---|---|
| **Core Task** | Compress 18-dimension engineered feature space and build a production-ready API |
| **Techniques Used** | StandardScaler, Principal Component Analysis (PCA), Random Forest Regressor |
| **Web Tech Stack** | FastAPI, Uvicorn, Jinja2 Templates, HTML5/CSS3 (Semantic Styles) |
| **Saved Artifact** | `models/model.joblib` (Contains the trained pipeline, feature names, and categorical metadata) |

---

## Project Structure

```
Week_03/
├── models/
│   └── model.joblib                 # Serialized model pipeline & metadata
├── notebooks/
│   └── week3_pca.ipynb              # Notebook implementing scaling, PCA, & pipeline creation
├── static/
│   ├── css/
│   │   └── style.css                # Custom styling for UI & Dashboard layout
│   └── images/
│       ├── cumulative_variance.png  # Explained variance ratio curve
│       ├── pca_loading_heatmap.png  # Heatmap of principal components loadings
│       └── scree_plot.png           # Scree plot for eigenvalue selection
├── templates/
│   ├── base.html                    # Common layout shell (navigation, footer)
│   ├── home.html                    # Welcome page
│   ├── dashboard.html               # Interactive static charts visualization
│   └── predict.html                 # Interactive Web UI for real-time inference
├── main.py                          # FastAPI web application entrypoint
├── requirements.txt                 # Project dependencies list
└── README.md                        # Project documentation
```

---

## Setup & Installation

Follow these steps to set up your environment, run the training pipeline, and launch the web server:

1. **Activate Virtual Environment:**
   ```bash
   # On Windows
   myenv\Scripts\activate
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train and Export the Model:**
   Open and run all cells in `notebooks/week3_pca.ipynb` to generate the updated model pipeline artifact:
   ```bash
   jupyter notebook notebooks/week3_pca.ipynb
   ```
   *(This exports the final pipeline into `models/model.joblib`)*

4. **Launch the FastAPI App:**
   ```bash
   uvicorn main:app --reload
   ```
   Now, open your browser and navigate to: `http://127.0.0.1:8000`

---

## Part 1 — Principal Component Analysis (PCA)

**Notebook:** `notebooks/week3_pca.ipynb`

### What was done

| Step | Description |
| --- | --- |
| **Feature Scaling** | Applied `StandardScaler` to bring all numeric features into the same scale for PCA |
| **Variance Analysis** | Generated Scree and Cumulative Variance plots to find the optimal number of dimensions |
| **Dimensionality Selection** | Chose **11 Principal Components** which successfully retain **95%** of the original variance |
| **Multicollinearity Fix** | Resolved high multicollinearity among power readings (Lagging/Leading) using PCA projection |
| **Pipeline Construction** | Built a robust `Pipeline([('scaler'), ('pca'), ('regressor')])` preventing leakage during inference |

### Visualizations Saved inside `static/images/`

* **Cumulative Explained Variance:** Visualizes how much total variance is captured as PCA dimensions increase.
* **PCA Loading Heatmap:** Reveals the feature combinations that make up each principal component.
* **Scree Plot:** Displays individual component eigenvalues.

---

## Part 2 — FastAPI Web Deployment & Dashboard

**Script:** `main.py` | **UI Templates:** `templates/`

A sleek web interface was developed with structural, modular layouts utilizing Jinja2 template inheritance (`base.html`):

### Key Pages

* **Home Page (`/`):** Introducing the UCI energy project workflow, describing how PCA compression optimizes the backend engine.
* **EDA Dashboard (`/dashboard`):** Showcases structural analytics and PCA plots generated during modeling directly inside a responsive grid.
* **Real-Time Inference (`/predict`):** Features a web form with dynamic validation where users can input real-time sensor metrics to receive instant kW-hour predictions.

---

## PCA Model Comparison Summary

| Pipeline Version | Number of Features | Test MAE | Test RMSE | Test R² Score |
| --- | --- | --- | --- | --- |
| **Original Baseline (Week 2)** | 18 | 0.342 | 1.053 | 0.9990 |
| **3 PCA Components** | 3 | 2.494 | 5.001 | 0.9776 |
| **11 PCA Components (95% Var)** | **11** | **1.547** | **3.059** | **0.9916** |

### Insights

* **Trade-off:** Compressing the input dimension from 18 to 11 features resulted in a minor drop in R² (from `0.9990` to `0.9916`), but it drastically streamlined the feature footprint, speeding up computing times and removing multicollinearity.
* **Generalization:** The PCA pipeline maintains outstanding robustness and is less prone to overfitting due to the elimination of noise features.

---

## Author

**Kiran Aslam**