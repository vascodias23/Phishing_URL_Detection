import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.metrics import confusion_matrix
import plotly.figure_factory as ff
import plotly.graph_objects as go


def plot_pca_explained_variance(pca, explained_variance=0.95):
    # Cumulative explained variance
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

    fig = px.line(
        x=range(1, len(cumulative_variance) + 1),
        y=cumulative_variance,
        markers=True,
        title="PCA Cumulative Explained Variance",
        labels={'x': 'Number of Components', 'y': 'Cumulative Explained Variance'}
    )
    fig.add_hline(y=explained_variance, line_dash="dash", line_color="red",
                  annotation_text="{:.2f}% Variance Threshold".format(explained_variance * 100),)
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
    fig.update_yaxes(range=[0.95, 1])
    fig.show()

def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    predicted_labels = ['Predicted Legitimate', 'Predicted Phishing']
    true_labels = ['True Legitimate', 'True Phishing']
    fig = ff.create_annotated_heatmap(
        cm, x=predicted_labels, y=true_labels,
        colorscale='Blues', showscale=True
    )
    fig.update_layout(title=title)
    fig.show()
