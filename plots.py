import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.metrics import confusion_matrix
import plotly.figure_factory as ff


def plot_pca_explained_variance(pca):
    # Cumulative explained variance
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

    fig = px.line(
        x=range(1, len(cumulative_variance) + 1),
        y=cumulative_variance,
        markers=True,
        title="PCA Cumulative Explained Variance",
        labels={'x': 'Number of Components', 'y': 'Cumulative Explained Variance'}
    )
    fig.add_hline(y=0.95, line_dash="dash", line_color="red",
                  annotation_text="95% Variance Threshold")
    fig.show()


def plot_lda_projection(X_lda, y, title="LDA Projection by Class"):
    df = pd.DataFrame({'LD1': X_lda.squeeze(), 'Label': y.reset_index(drop=True)})
    fig = px.histogram(
        df, x='LD1', color='Label',
        title=title,
        nbins=100, barmode='overlay',
        opacity=0.7
    )
    fig.show()


def plot_performance_metrics(results_df):
    # Melt the DataFrame for easier plotting
    melted_df = pd.melt(results_df, id_vars=['Method', 'Classifier'],
                        value_vars=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                        var_name='Metric', value_name='Value')

    fig = px.bar(
        melted_df, x='Method', y='Value', color='Classifier',
        facet_col='Metric', barmode='group',
        title="Classifier Performance Across Methods and Metrics"
    )
    fig.update_layout(height=600, width=1200)
    fig.show()

def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    labels = ['Legitimate', 'Phishing']
    fig = ff.create_annotated_heatmap(
        cm, x=labels, y=labels,
        colorscale='Blues', showscale=True
    )
    fig.update_layout(title=title)
    fig.show()


def plot_decision_boundary(X, y, title):
    df = pd.DataFrame({'PC1': X[:, 0], 'PC2': X[:, 1], 'Label': y})
    fig = px.scatter(
        df, x='PC1', y='PC2', color='Label',
        title=title, opacity=0.5
    )

    # Calculate class means
    means = df.groupby('Label').mean()
    fig.add_trace(px.scatter(
        means, x='PC1', y='PC2',
        symbol=means.index,
        symbol_sequence=['x', 'x'],
        size=[20, 20],
        color_discrete_sequence=['black', 'black']
    ).data[0])

    fig.show()