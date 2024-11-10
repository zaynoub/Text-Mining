import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import TruncatedSVD
import scipy.sparse
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Charger les données et la matrice TF-IDF
df = pd.read_csv('data/cleaned_data.csv')
X_tfidf = scipy.sparse.load_npz('data/X_tfidf.npz')  # Charger la matrice sparse

# Méthode Elbow
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

# Méthode de la silhouette
silhouette_scores = []
for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_tfidf)
    score = silhouette_score(X_tfidf, labels)
    silhouette_scores.append(score)

plt.figure(figsize=(8, 4))
plt.plot(range(2, 10), silhouette_scores, marker='o')
plt.xlabel('Nombre de Clusters')
plt.ylabel('Score de silhouette')
plt.title('Méthode de la silhouette pour trouver le nombre optimal de clusters')
plt.savefig('visualizations/silhouette_method.png')
plt.show()

# Méthode Davies-Bouldin
davies_bouldin_scores = []
for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_tfidf)
    score = davies_bouldin_score(X_tfidf.toarray(), labels)
    davies_bouldin_scores.append(score)

plt.figure(figsize=(8, 4))
plt.plot(range(2, 10), davies_bouldin_scores, marker='o')
plt.xlabel('Nombre de Clusters')
plt.ylabel('Score Davies-Bouldin')
plt.title('Méthode Davies-Bouldin pour trouver le nombre optimal de clusters')
plt.savefig('visualizations/davies_bouldin_method.png')
plt.show()
