import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

from plots import plot_lda_projection, plot_pca_explained_variance, plot_decision_boundary, plot_performance_metrics, plot_confusion_matrix


def pre_processing(heatmap=False, verbose=False, box_plots=False, histograms=False):
    df = pd.read_csv('data/PhiUSIIL_Phishing_URL_Dataset.csv')

    # This will select only numeric (non-categorical) columns (int, float)
    y = df['label']
    df_numeric = df.select_dtypes(include=['int64', 'float64']).drop(columns='label')
    if verbose:
        print(df_numeric.info())
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
            fig = px.histogram(df_numeric[feat], nbins=500, title=f"{feat} Distribution",)
            fig.show()

    # Plot boxplots for all features to visualize distributions
    if box_plots:
        for feat in df_numeric.columns:
            fig = px.box(df_numeric[feat], title=f"{feat} Boxplot", )
            fig.show()

    scaler = StandardScaler()
    # Fit and transform the non-categorical features
    scaled_features = scaler.fit_transform(df_numeric)
    # Convert back to DataFrame for convenience
    df_scaled = pd.DataFrame(scaled_features, columns=df_numeric.columns)

    if verbose:
        print(df_scaled.describe())

    return df_scaled, y


def apply_pca(X_train, X_test, n_components=0.95):
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    return X_train_pca, X_test_pca, pca


def apply_lda(X_train, X_test, y_train, n_components=1):
    lda = LDA(n_components=n_components)
    X_train_lda = lda.fit_transform(X_train, y_train)
    X_test_lda = lda.transform(X_test)
    return X_train_lda, X_test_lda, lda


def apply_pca_lda(X_train, X_test, y_train, n_components_pca=0.95, n_components_lda=1):
    # Apply PCA first
    X_train_pca, X_test_pca, pca = apply_pca(X_train, X_test, n_components_pca)
    # Then apply LDA on PCA-transformed data
    X_train_pca_lda, X_test_pca_lda, lda = apply_lda(X_train_pca, X_test_pca, y_train, n_components_lda)
    return X_train_pca_lda, X_test_pca_lda, (pca, lda)


def minimum_distance_classifier(X_train, y_train, X_test):
    # Convert inputs to numpy arrays to handle both DataFrames and numpy arrays
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    classes = np.unique(y_train)
    # Calculate the mean for each class
    means = {c: np.mean(X_train[y_train == c], axis=0) for c in classes}
    # Predict the class with minimum distance for each test sample
    y_pred = [classes[np.argmin([np.linalg.norm(x - means[c]) for c in classes])] for x in X_test]
    return np.array(y_pred)


def evaluate_model(y_true, y_pred, title):
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1-Score': f1_score(y_true, y_pred)
    }
    print(f"\n{title} Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    plot_confusion_matrix(y_true, y_pred, title)
    return metrics


def main():
    df_scaled, y = pre_processing(heatmap=True, verbose=True, histograms=False, box_plots=False)
    df_scaled.to_csv('data/scaled_data.csv', index=False)
    print("Scaled data saved to 'data/scaled_data.csv'.")

    # Split between train and test data (cross-validation)
    X_train, X_test, y_train, y_test = train_test_split(df_scaled, y, test_size=0.2, stratify=y, random_state=42)

    # --- Feature Reduction ---
    # 1. PCA only
    X_train_pca, X_test_pca, pca = apply_pca(X_train, X_test)

    # 2. LDA only (on original scaled data)
    X_train_lda, X_test_lda, lda = apply_lda(X_train, X_test, y_train)

    # 3. PCA + LDA
    X_train_pca_lda, X_test_pca_lda, _ = apply_pca_lda(X_train, X_test, y_train)

    plot_pca_explained_variance(pca)
    plot_lda_projection(X_train_lda, y_train)
    plot_decision_boundary(X_train_pca[:, :2], y_train, "MDC Decision Boundary (PCA Components)")

    # --- Classifiers ---
    # Define datasets to evaluate
    datasets = {
        'Original': (X_train, X_test),
        'PCA': (X_train_pca, X_test_pca),
        'LDA': (X_train_lda, X_test_lda),
        'PCA+LDA': (X_train_pca_lda, X_test_pca_lda)
    }

    # Store results
    results = []

    for name, (X_tr, X_te) in datasets.items():
        # Minimum Distance Classifier (MDC)
        y_pred_mdc = minimum_distance_classifier(X_tr, y_train, X_te)
        metrics_mdc = evaluate_model(y_test, y_pred_mdc, f"{name} - MDC")
        results.append({'Method': name, 'Classifier': 'MDC', **metrics_mdc})

        # Fisher's LDA Classifier
        if name != 'LDA':  # Avoid redundancy if LDA is already the transformation
            lda_clf = LDA()
            lda_clf.fit(X_tr, y_train)
            y_pred_lda = lda_clf.predict(X_te)
            metrics_lda = evaluate_model(y_test, y_pred_lda, f"{name} - LDA Classifier")
            results.append({'Method': name, 'Classifier': 'LDA', **metrics_lda})

    # Display results as DataFrame
    results_df = pd.DataFrame(results)
    plot_performance_metrics(results_df)
    print("\nFinal Results:")
    print(results_df.to_string(index=False))

if __name__ == "__main__" :
    main()