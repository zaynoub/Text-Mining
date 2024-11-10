import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.sparse
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_samples, silhouette_score, confusion_matrix
import umap.umap_ as umap  # Assurez-vous que l'import est correct
import itertools

# Fonction pour afficher et enregistrer les graphes de silhouette
def plot_silhouette(X, labels, model_name):
    silhouette_avg = silhouette_score(X, labels)
    sample_silhouette_values = silhouette_samples(X, labels)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    y_lower = 10
    n_clusters = len(np.unique(labels))
    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[labels == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          alpha=0.7)
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10

    ax1.set_title(f"Graphique de silhouette pour {model_name}")
    ax1.set_xlabel("Valeur du coefficient de silhouette")
    ax1.set_ylabel("Cluster")
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    plt.savefig(f'visualizations/silhouette_{model_name}.png')  # Enregistrer l'image
    plt.show()

# Fonction pour afficher et enregistrer la matrice de confusion entre les clusters
def plot_confusion_matrix(y_true, y_pred, model1, model2):
    matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Matrice de confusion entre {model1} et {model2}")
    plt.xlabel(f"Clusters de {model2}")
    plt.ylabel(f"Clusters de {model1}")
    plt.savefig(f'visualizations/confusion_matrix_{model1}_vs_{model2}.png')  # Enregistrer l'image
    plt.show()

# Entraînement des modèles et visualisation
def main():
    # Charger les données nettoyées
    df = pd.read_csv('data/cleaned_data.csv')
    
    # Charger la matrice sparse TF-IDF
    X_tfidf = scipy.sparse.load_npz('data/X_tfidf.npz')
    print("Matrice TF-IDF chargée avec succès.")

    # Sous-échantillonner la matrice TF-IDF pour l'entraînement et la visualisation
    sample_size = 10000  # Ajuster selon la mémoire disponible
    sample_indices = np.random.choice(X_tfidf.shape[0], sample_size, replace=False)
    X_tfidf_sample = X_tfidf[sample_indices].toarray()
    df_sample = df.iloc[sample_indices]

    # Entraînement KMeans sur le sous-échantillon
    kmeans = KMeans(n_clusters=4, random_state=42)
    df_sample['cluster_kmeans'] = kmeans.fit_predict(X_tfidf_sample)
    print("KMeans entraîné sur un sous-échantillon.")

    # Afficher le graphe de silhouette pour KMeans
    plot_silhouette(X_tfidf_sample, df_sample['cluster_kmeans'], "KMeans")

    # Entraînement DBSCAN sur le sous-échantillon
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    df_sample['cluster_dbscan'] = dbscan.fit_predict(X_tfidf_sample)
    print("DBSCAN entraîné sur un sous-échantillon.")
    
    # Vérifier si DBSCAN a plus d'un cluster pour calculer le coefficient de silhouette
    if len(np.unique(df_sample['cluster_dbscan'])) > 1:
        plot_silhouette(X_tfidf_sample, df_sample['cluster_dbscan'], "DBSCAN")

    # Entraînement Agglomerative Clustering sur le sous-échantillon
    agglo = AgglomerativeClustering(n_clusters=4)
    df_sample['cluster_agglo'] = agglo.fit_predict(X_tfidf_sample)
    print("Agglomerative Clustering entraîné sur un sous-échantillon.")

    # Afficher le graphe de silhouette pour Agglomerative Clustering
    plot_silhouette(X_tfidf_sample, df_sample['cluster_agglo'], "Agglomerative Clustering")

    # Visualisation de la matrice de confusion entre les modèles
    plot_confusion_matrix(df_sample['cluster_kmeans'], df_sample['cluster_agglo'], "KMeans", "Agglomerative Clustering")
    plot_confusion_matrix(df_sample['cluster_kmeans'], df_sample['cluster_dbscan'], "KMeans", "DBSCAN")
    plot_confusion_matrix(df_sample['cluster_agglo'], df_sample['cluster_dbscan'], "Agglomerative Clustering", "DBSCAN")

    print("Entraînement et visualisations terminés.")

if __name__ == "__main__":
    main()
