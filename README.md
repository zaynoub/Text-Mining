# Text-Mining
# Community Detection in Twitter Data

## Description

This project focuses on analyzing a large set of tweets from Twitter in order to automatically detect user communities. These communities are formed based on common interests or similar behaviors, and their identification is crucial for understanding social dynamics and emerging trends on the platform.

The project uses text mining techniques to preprocess and clean the tweets, which include tasks such as the removal of unwanted elements (URLs, mentions, hashtags, special characters), stopword customization, and lemmatization. After the preprocessing phase, clustering algorithms (KMeans, DBSCAN, Agglomerative Clustering) are applied to group users into thematic communities. The DBSCAN algorithm was found to be particularly effective in managing diverse cluster structures and identifying outliers.

## Objectives
- Preprocessing Twitter data for text mining.
- Customizing stopwords for better fitting social media data.
- Applying clustering techniques to detect thematic communities on Twitter.
- Using algorithms like KMeans, DBSCAN, and Agglomerative Clustering to segment users.

## Data

The data used for this project consists of a large collection of tweets gathered from the Twitter platform. The dataset includes user posts along with metadata such as the text content, user IDs, and timestamps. This data is essential for analyzing and clustering user communities based on their shared interests or behaviors.

You can access the dataset used for this project through the following link:  
[Link to Data](https://www.kaggle.com/datasets/kazanova/sentiment140)
