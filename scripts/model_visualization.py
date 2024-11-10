import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import TruncatedSVD
import scipy.sparse
import umap.umap_ as umap  # Assurez-vous que l'import est correct
from sklearn.cluster import KMeans

# Charger les données et la matrice TF-IDF
df = pd.read_csv('data/cleaned_data.csv')
X_tfidf = scipy.sparse.load_npz('data/X_tfidf.npz')  # Charger la matrice sparse

# Appliquer KMeans pour assigner les clusters
kmeans = KMeans(n_clusters=4, random_state=42)
df['cluster'] = kmeans.fit_predict(X_tfidf)

# Visualisation de la méthode Elbow
inertias = []
K = range(1, 10)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_tfidf)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(K, inertias, marker='o')
plt.xlabel('Nombre de Clusters')
plt.ylabel("Inertie")
plt.title("Méthode Elbow pour trouver le nombre optimal de clusters")
plt.savefig('visualizations/elbow_method.png')
plt.show()

# Sous-échantillonner la matrice TF-IDF pour la visualisation
sample_size = 50000  # Ajuster ce nombre en fonction de la mémoire disponible
indices = np.random.choice(X_tfidf.shape[0], sample_size, replace=False)
X_tfidf_sample = X_tfidf[indices].toarray()  # Convertir le sous-échantillon en dense
clusters_sample = df['cluster'].iloc[indices]

# Réduction de dimension avec TruncatedSVD
svd = TruncatedSVD(n_components=50, random_state=42)
X_svd_sample = svd.fit_transform(X_tfidf_sample)

# Application de UMAP
umap_reducer = umap.UMAP(n_components=2, random_state=42)
X_umap = umap_reducer.fit_transform(X_svd_sample)

df_umap = pd.DataFrame(X_umap, columns=['x', 'y'])
df_umap['cluster'] = clusters_sample.values

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_umap, x='x', y='y', hue='cluster', palette='viridis', s=60)
plt.title("Visualisation des Clusters avec TruncatedSVD et UMAP (sous-échantillon)")
plt.savefig('visualizations/umap_clusters.png')
plt.show()
