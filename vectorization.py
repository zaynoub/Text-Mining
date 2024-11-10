import pandas as pd
import joblib
import scipy.sparse
from sklearn.feature_extraction.text import TfidfVectorizer

# Vectorisation des données
def main():
    # Charger les données nettoyées
    df = pd.read_csv('data/cleaned_data.csv')

    # Vérifier les valeurs manquantes et les remplacer par des chaînes vides
    if df['lemmatized_text'].isnull().any():
        print("Des valeurs manquantes ont été détectées dans 'lemmatized_text' et remplacées par des chaînes vides.")
        df['lemmatized_text'].fillna('', inplace=True)

    # Initialiser le vectoriseur TF-IDF
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    X_tfidf = tfidf_vectorizer.fit_transform(df['lemmatized_text'])

    # Afficher la forme de la matrice TF-IDF
    print("Shape of TF-IDF features:", X_tfidf.shape)

    # Enregistrer le modèle TF-IDF pour réutilisation ultérieure
    joblib.dump(tfidf_vectorizer, 'data/tfidf_vectorizer.pkl')
    print("Modèle TF-IDF enregistré sous 'data/tfidf_vectorizer.pkl'.")

    # Sauvegarder la matrice sparse TF-IDF dans un fichier compressé
    scipy.sparse.save_npz('data/X_tfidf.npz', X_tfidf)
    print("Matrice TF-IDF enregistrée sous 'data/X_tfidf.npz'.")

if __name__ == "__main__":
    main()
