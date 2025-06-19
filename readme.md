# Pattern Recognition for Phishing URL Detection

An educational project demonstrating basic pattern‐recognition techniques (PCA, LDA, k-NN, SVM, etc.) on a phishing‐URL dataset inspired by the PhiUSIIL framework :contentReference[oaicite:0]{index=0}.

## Files

- **main.py** – loads the PhiUSIIL dataset, removes correlated features, scales data, applies PCA/LDA transforms, and runs 5-fold classifiers (MDC, LDA, k-NN, SGD, SVM). :contentReference[oaicite:1]{index=1}  
- **plots.py** – helper functions to visualize explained variance, LDA projection, performance metrics, and confusion matrices. :contentReference[oaicite:2]{index=2}
- **data/** – place `PhiUSIIL_Phishing_URL_Dataset.csv` here (available on the supplementary materials of the PhiUSIIL paper).

## Getting Started

1. **Clone** this repo  
2. **Install** dependencies:  
   ```bash
   pip install pandas numpy scikit-learn plotly
