import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

from plots import plot_lda_projection, plot_pca_explained_variance, plot_performance_metrics, plot_confusion_matrix

# Adapted from a previous 'pre_processing' function
def data_analysis(heatmap=False, verbose=False, box_plots=False, histograms=False):
    df = pd.read_csv('data/PhiUSIIL_Phishing_URL_Dataset.csv')

    # This will select only numeric (non-categorical) columns (int, float)
    y = df['label']
    df_numeric = df.select_dtypes(include=['int64', 'float64']).drop(columns='label')
    if verbose:
        print(df_numeric.info())
        print(df_numeric.describe())

    if verbose:
        print(df_numeric.describe())

    # Compute the correlation matrix
    corr_matrix = df_numeric.corr()

    # Interactive heatmap to visualize highly correlated features
    if heatmap:
        fig = px.imshow(corr_matrix,
                        text_auto=True,
                        aspect='auto',
                        title='Correlation Matrix',
                        )
        fig.show()

    # Identify highly correlated feature pairs above a certain threshold
    threshold = 0.9
    correlated_features = set()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                colname_i = corr_matrix.columns[i]
                colname_j = corr_matrix.columns[j]
                correlated_features.add((colname_i, colname_j, corr_matrix.iloc[i, j]))

    # Display highly correlated feature pairs
    if verbose:
        for feat1, feat2, corr_value in correlated_features:
            print(f"{feat1} and {feat2} have a correlation of {corr_value:.2f}")

    # Plot histograms for all features to visualize distributions
    if histograms:
        for feat in df_numeric.columns:
            fig = px.histogram(df_numeric[feat], nbins=500, title=f"{feat} Distribution", )
            fig.show()

    # Plot boxplots for all features to visualize distributions
    if box_plots:
        for feat in df_numeric.columns:
            fig = px.box(df_numeric[feat], title=f"{feat} Boxplot", )
            fig.show()

    return


def remove_correlated_features(X_train, y_train, threshold=0.95):
    corr_matrix = X_train.corr().abs()
    
    correlated_pairs = []
    columns = X_train.columns
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            if corr_matrix.iloc[i, j] > threshold:
                correlated_pairs.append((columns[i], columns[j]))

    features_to_drop = set()
    for feat1, feat2 in correlated_pairs:
        # Calculate correlation with target
        corr1 = abs(np.corrcoef(X_train[feat1], y_train)[0, 1])
        corr2 = abs(np.corrcoef(X_train[feat2], y_train)[0, 1])

        if corr1 > corr2:
            features_to_drop.add(feat2)
        else:
            features_to_drop.add(feat1)

    # Ensure features exist in the DataFrame
    valid_drops = [f for f in features_to_drop if f in X_train.columns]
    return X_train.drop(columns=valid_drops), valid_drops


def apply_transformations(X_train, X_test, y_train, n_components_pca=0.95, n_components_lda=1):
    # PCA
    pca = PCA(n_components=n_components_pca)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    # LDA (on original data)
    lda = LDA(n_components=n_components_lda)
    X_train_lda = lda.fit_transform(X_train, y_train)
    X_test_lda = lda.transform(X_test)

    # PCA + LDA
    lda_pca = LDA(n_components=n_components_lda)
    X_train_pca_lda = lda_pca.fit_transform(X_train_pca, y_train)
    X_test_pca_lda = lda_pca.transform(X_test_pca)

    return {
        'Original': (X_train, X_test),
        'PCA': (X_train_pca, X_test_pca),
        'LDA': (X_train_lda, X_test_lda),
        'PCA+LDA': (X_train_pca_lda, X_test_pca_lda)
    }


def minimum_distance_classifier(X_train, y_train, X_test):
    # Convert inputs to numpy arrays to handle both DataFrames and numpy arrays
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    classes = np.unique(y_train)
    # Calculate the mean for each class
    means = {c: np.mean(X_train[y_train == c], axis=0) for c in classes}
    # Predict the class with minimum distance for each test sample (Euclidean)
    y_pred = [classes[np.argmin([np.linalg.norm(sample - means[c]) for c in classes])] for sample in X_test]
    return np.array(y_pred)


def evaluate_model(y_true, y_pred, title, visualize=False):
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1-Score': f1_score(y_true, y_pred)
    }
    print(f"\n{title} Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    if visualize:
        plot_confusion_matrix(y_true, y_pred, title)
    return metrics


def main():
    data_analysis(verbose=True)

    df = pd.read_csv('data/PhiUSIIL_Phishing_URL_Dataset.csv')
    y = df['label']
    X = df.select_dtypes(include=['int64', 'float64']).drop(columns='label')

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train_raw, X_test_raw = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # 1. Remove correlated features (using training data only)
        X_train_reduced, dropped_features = remove_correlated_features(X_train_raw, y_train)
        X_test_reduced = X_test_raw.drop(columns=dropped_features)

        # 2. Scale data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_reduced)
        X_test_scaled = scaler.transform(X_test_reduced)

        # 3. Apply transformations
        datasets = apply_transformations(X_train_scaled, X_test_scaled, y_train)

        # 4. Evaluate classifiers for each dataset
        for dataset_name, (X_tr, X_te) in datasets.items():
            # Minimum Distance Classifier
            y_pred_mdc = minimum_distance_classifier(X_tr, y_train, X_te)
            metrics_mdc = {
                'Accuracy': accuracy_score(y_test, y_pred_mdc),
                'Precision': precision_score(y_test, y_pred_mdc),
                'Recall': recall_score(y_test, y_pred_mdc),
                'F1': f1_score(y_test, y_pred_mdc)
            }
            results.append({
                'Fold': fold,
                'Dataset': dataset_name,
                'Classifier': 'MDC',
                **metrics_mdc
            })

            # LDA Classifier (skip for LDA dataset to avoid redundancy)
            if dataset_name != 'LDA':
                lda_clf = LDA()
                lda_clf.fit(X_tr, y_train)
                y_pred_lda = lda_clf.predict(X_te)
                metrics_lda = {
                    'Accuracy': accuracy_score(y_test, y_pred_lda),
                    'Precision': precision_score(y_test, y_pred_lda),
                    'Recall': recall_score(y_test, y_pred_lda),
                    'F1': f1_score(y_test, y_pred_lda)
                }
                results.append({
                    'Fold': fold,
                    'Dataset': dataset_name,
                    'Classifier': 'LDA',
                    **metrics_lda
                })

    # Aggregate and display results
    results_df = pd.DataFrame(results)
    results_df.round(3).to_csv('results/full_results.csv', index=False)
    print("Full results saved to 'results/full_results.csv'.")

    summary = results_df.groupby(['Dataset', 'Classifier']).agg(['mean', 'std'])

    # Flatten multi-level column headers for CSV
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary.round(3).to_csv('results/results_summary.csv', index=True)
    print("Results saved to 'results/results_summary.csv'.")

    with open('results/results_summary.txt', 'w') as f:
        f.write("=== Phishing URL Detection Results ===\n\n")
        f.write("Aggregated Metrics (Mean ± Std):\n")
        f.write(summary.to_string())
        f.write("\n\n=== Full Results (All Folds) ===\n")
        f.write(results_df.to_string(index=False))
    print("Text summary saved to 'results/results_summary.txt'.")


if __name__ == "__main__":
    main()
