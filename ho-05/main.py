import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
"""
Auxiliary Functions
"""


# Punctuation Removal
def remove_punctuation(text):
  return re.sub(r"[^\w\s\d]|\n", '', text)


"""
Data
"""
# Open file, create normalized database
file = open("../datasets/headlines.txt", "r")
data = []
for line in file:
  data.append(remove_punctuation(line).lower())
"""
Embeddings
"""
#One-hot Encoding
vectorizer = CountVectorizer(binary=True)
oh_embedding = vectorizer.fit_transform(data)

#Count Vectors
vectorizer = CountVectorizer()
cv_embedding = vectorizer.fit_transform(data)

#TF-IDF
vectorizer = TfidfVectorizer()
tf_embedding = vectorizer.fit_transform(data)

#n-grams (2-grams)
vectorizer = CountVectorizer(ngram_range=(2, 2))
ng_embedding = vectorizer.fit_transform(data)

#Co-occurrence Vectors (Context Window = 1)
vectorizer = CountVectorizer()
vectorizer.fit(data)
vocab = vectorizer.vocabulary_
co_embedding = []
for i in range(len(vocab)):
  co_embedding.append([0] * len(vocab))

for i in range(len(vocab)):
  co_embedding[i][i] = 1

for line in data:
  l = line.split()
  for i in range(1, len(l)):
    if (l[i - 1] in vocab and l[i] in vocab):
      co_embedding[vocab[l[i - 1]]][vocab[l[i]]] += 1

#Word2Vec
words = []
for line in data:
  words.append(line.split())

model = Word2Vec(sentences=words,
                 vector_size=100,
                 window=5,
                 min_count=1,
                 workers=4)
wv_embedding = model.wv.vectors
"""
Text Similarity
"""

#Jaccard

#Manhattan

#Euclidean

#Minkowski com p=3

#Cosine Similarity
