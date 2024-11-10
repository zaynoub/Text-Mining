import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Télécharger les ressources nécessaires de NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialisation du lemmatizer et des stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Stopwords personnalisés supplémentaires
custom_stopwords = {
    'theyre', 'shes', 'hes', 'thm', 'wat', 'wut', 'hoo', 'thats', 'dis', 
    'dat', 'thse', 'wuz', 'wer', 'bein', 'havent', 'havnt', 'hav', 'haz', 
    'doin', 'dosent', 'didnt', 'cuz', 'bcuz', 'becuz', 'coz', 'bcoz', 'get', 
    'got', 'shoulda', 'becoz', 'til', 'till', 'az', 'whl', 'whle', 'abt', 
    'bout', 'btwn', 'thru', 'durin', 'abv', 'frm', 'ovr', 'wen', 'hw', 
    'othr', 'shld', 'shouldnt', 'id', 'tbh', 'bff', 'btw', 'smh', 'fomo', 
    'thx', 'lmk', 'fml', 'plz', 'fyi', 'na', 'im', 'dont', 'cant', 'gotta', 
    'wanna', 'gonna', 'aint', 'lol', 'lmao', 'u', 'ur', 'idk', 'dm', 'like', 
    'go', 'day', 'good', 'great', 'really', 'time', 'today', 'back', 'one', 
    'see', 'omg', 'hey', 'hi', 'hello', 'nah', 'yeah', 'yep', 'nope', 'pls', 
    'haha', 'hehe', 'thanks', 'thank', 'ok', 'okay', 'alright', 'amp', 'rt', 
    'oh', 'ah', 'uh'
}
all_stopwords = stop_words.union(custom_stopwords)

# Fonction de nettoyage de texte avec NLTK
def clean_text(text):
    # Supprimer les URL, mentions et caractères spéciaux
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    return text.lower()

# Fonction de tokenisation et lemmatisation avec NLTK
def lemmatize_text(text):
    tokens = word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in all_stopwords and len(token) > 1]
    return ' '.join(lemmatized_tokens)

# Chargement et nettoyage des données
def main():
    file_path = "data/raw_data.csv"  # Chemin vers le fichier brut
    df = pd.read_csv(file_path, encoding="ISO-8859-1", names=["polarity", "id", "date", "query", "user", "text"])
    
    # Appliquer le nettoyage et la lemmatisation
    df['cleaned_text'] = df['text'].apply(clean_text)
    df['lemmatized_text'] = df['cleaned_text'].apply(lemmatize_text)
    
    # Sauvegarder les données nettoyées
    df.to_csv('data/cleaned_data.csv', index=False)
    print("Nettoyage des données terminé et fichier sauvegardé.")

if __name__ == "__main__":
    main()
