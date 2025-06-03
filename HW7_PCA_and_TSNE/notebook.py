#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import lemmagen.lemmatizer
from lemmagen.lemmatizer import Lemmatizer
from vispy import scene
from vispy.scene import visuals


# In[2]:


df = pd.read_json('rtvslo_keywords.json')
df.head()


# In[3]:


concatenated_keywords = df['gpt_keywords'].map(lambda x: '$split$'.join(x).lower())
    
lemmatizer = Lemmatizer(dictionary=lemmagen.DICTIONARY_SLOVENE)

def comma_tokenizer(s):
    phrases = s.split('$split$')
    lemmatized_phrases = []
    for phrase in phrases:
        words = phrase.split()
        lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
        lemmatized_phrase = ' '.join(lemmatized_words)
        lemmatized_phrases.append(lemmatized_phrase)
    return lemmatized_phrases

vectorizer = TfidfVectorizer(min_df=20, tokenizer=comma_tokenizer, lowercase=False)

X = vectorizer.fit_transform(concatenated_keywords)
tfidf_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())


# In[4]:


# Write column names to a file
tfidf_df.columns.to_series().to_csv('column_names.csv', index=False)


# In[5]:


from sklearn.preprocessing import MinMaxScaler

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Fit the scaler and transform the data
tfidf_df_normalized = pd.DataFrame(scaler.fit_transform(tfidf_df), columns=tfidf_df.columns)


# In[6]:


tfidf_df_normalized.head()


# In[13]:


pca = PCA(n_components=2000)
pca.fit(tfidf_df_normalized)


# In[14]:


# Get explained variance ratios (lambdas)
lambdas = pca.explained_variance_ratio_ * 100
cumulative_lambdas = np.cumsum(lambdas)

# Plot scree plot
plt.plot(np.arange(1, len(lambdas) + 1), lambdas)
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Proportion of Variance Explained')

# Plot cumulative sum of lambdas
plt.plot(np.arange(1, len(cumulative_lambdas) + 1), cumulative_lambdas)
plt.title('Cumulative Sum of Lambdas')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Proportion of Variance Explained')

plt.tight_layout()
plt.show()


# In[9]:


print(lambdas[0])

print(1/2998)


# In[10]:


pca = PCA(n_components=3)
X_pca = pca.fit(tfidf_df)


# In[61]:


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm


def biplot_3d(scores, loadings, pvars, features=None):
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, projection='3d')
    xs = scores[:,0]
    ys = scores[:,1]
    zs = scores[:,2]
    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())
    scalez = 1.0/(zs.max() - zs.min())

    #arrows = loadings * np.abs(scores).max(axis=0)
    arrows = loadings *2#* np.std(scores, axis=0)  # Scale arrows by standard deviation

    print(loadings[0])


    ax.scatter(xs * scalex, ys * scaley, zs * scalez)

    # empirical formula to determine arrow width
    # features as arrows
    max_indices = np.argpartition(arrows, -3, axis=0)[-2:].flatten()
    min_indices = np.argpartition(arrows, 3, axis=0)[:2].flatten()


    print(max_indices)
    print(min_indices)

    # features as arrows
    for i in set(max_indices.tolist() + min_indices.tolist()):
        print(i)
        ax.quiver(0, 0, 0, *arrows[i], color='g', alpha=1)
        ax.text(*(arrows[i] * 1.05), features[i],
                 ha='center', va='center')

    # axis labels
    #for i, axis in enumerate('xy'):
    #    getattr(plt, f'{axis}ticks')([])
    #    getattr(plt, f'{axis}label')(f'PC{i + 1} ({pvars[i]:.2f}%)')


# Assuming that your PCA transformed data is stored in X_pca and the PCA model in pca
biplot_3d(X_pca[:, :3],pca.components_[:3].T, pvars = pca.explained_variance_ratio_[:2] * 100, features=tfidf_df.columns)
plt.show()

