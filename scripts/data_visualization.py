import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
from collections import Counter
from wordcloud import WordCloud

# Charger les données nettoyées
df = pd.read_csv('data/cleaned_data.csv')

# Calcul de la longueur de chaque tweet
df['text_length'] = df['text'].apply(len)

# Distribution de la longueur des tweets
plt.figure(figsize=(10, 6))
sns.histplot(df['text_length'], bins=30, kde=True)
plt.title("Distribution de la Longueur des Tweets")
plt.xlabel("Longueur du Tweet")
plt.ylabel("Nombre de Tweets")
plt.savefig('visualizations/tweet_length_distribution.png')
plt.show()

# Top 10 utilisateurs avec le plus de tweets
top_users = df['user'].value_counts().head(10)

plt.figure(figsize=(10, 6))
sns.barplot(y=top_users.index, x=top_users.values)
plt.title("Top 10 Utilisateurs avec le Plus de Tweets")
plt.xlabel("Nombre de Tweets")
plt.ylabel("Utilisateur")
plt.savefig('visualizations/top_users.png')
plt.show()

# Extraction et visualisation des hashtags
def extract_hashtags(text):
    hashtags = re.findall(r'#\w+', text)
    return [hashtag[1:] for hashtag in hashtags]  # Supprime le '#' et garde seulement les mots

df['hashtags'] = df['text'].apply(extract_hashtags)
all_hashtags = [hashtag for hashtags_list in df['hashtags'] for hashtag in hashtags_list]
hashtag_counts = Counter(all_hashtags)

hashtag_df = pd.DataFrame(hashtag_counts.items(), columns=['Hashtag', 'Frequency'])
hashtag_df = hashtag_df.sort_values(by='Frequency', ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x='Frequency', y='Hashtag', data=hashtag_df, palette='viridis')
plt.title('Top 10 des Hashtags les Plus Fréquents')
plt.xlabel('Fréquence')
plt.ylabel('Hashtags')
plt.savefig('visualizations/top_hashtags.png')
plt.show()

# Les mots les plus fréquents dans les tweets
all_words = ' '.join(df['cleaned_text']).split()
common_words = Counter(all_words).most_common(30)
common_words_df = pd.DataFrame(common_words, columns=['Word', 'Frequency'])

plt.figure(figsize=(10, 6))
sns.barplot(data=common_words_df, x='Frequency', y='Word')
plt.title("Les 30 Mots les Plus Fréquents dans les Tweets")
plt.savefig('visualizations/common_words.png')
plt.show()

# Nuage de mots
word_freq = dict(common_words)
wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate_from_frequencies(word_freq)

plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Nuage de Mots des Tweets")
plt.savefig('visualizations/wordcloud.png')
plt.show()
