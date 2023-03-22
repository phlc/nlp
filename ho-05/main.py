import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
import seaborn as sn
import pandas as pd

"""
Auxiliary Functions
"""
#Punctuation Removal
def remove_punctuation(text):
  return re.sub(r"[^\w\s\d]|\n", '', text)

#Generate and Save Heatmap
def heatmap(array, name):
  df_cm = pd.DataFrame(array)

  
  svm = sn.heatmap(df_cm, annot=True, square=True, linecolor='white', linewidths=1)
  
  figure = svm.get_figure()    
  figure.savefig(f'{name}.png', dpi=100)
  figure.clf()

"""
Text Similarity
"""
#Jaccard
def jaccard(data):
  matrix = []
  for i in range(len(data)):
    matrix.append([0]*len(data))
    
  for i in range(len(data)):
    words_i = data[i].split()
    for j in range(i, len(data)):
      words_j = data[j].split()
      intersection = 0
      for word in words_i:
        if(word in words_j):
          intersection += 1
      matrix[i][j] = matrix[j][i] =  intersection / (len(words_i) + len(words_j) - intersection)
      
  return matrix

##Minkowski
def minkowski(data, p):
  matrix = []
  for i in range(len(data)):
    matrix.append([0]*len(data))

  for i in range(len(data)):
    for j in range(i, len(data)):
      matrix[i][j] = matrix[j][i] = round(pow(sum(pow(abs(k-l),p) for k, l in zip(data[i], data[j])),1/p),2)

  return matrix

#Cosine Similarity
def cosine_similarity(data):
  matrix = []
  for i in range(len(data)):
    matrix.append([0]*len(data))

  for i in range(len(data)):
    for j in range(i, len(data)):
      num = sum(k*l for k, l in zip(data[i], data[j]))
      fact1 = sum(k*k for k in data[i])
      fact2 = sum(k*k for k in data[j])
                                     
      matrix[i][j] = matrix[j][i] =  round(num/(pow(fact1,1/2)*pow(fact2,1/2)),2)

  return matrix
  
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
oh_embedding = vectorizer.fit_transform(data).toarray()

#Count Vectors
vectorizer = CountVectorizer()
cv_embedding = vectorizer.fit_transform(data).toarray()

#TF-IDF
vectorizer = TfidfVectorizer()
tf_embedding = vectorizer.fit_transform(data).toarray()

#n-grams (2-grams)
vectorizer = CountVectorizer(ngram_range=(2, 2))
ng_embedding = vectorizer.fit_transform(data).toarray()

#Co-occurrence Vectors (Context Window = 1)
vectorizer = CountVectorizer()
vectorizer.fit(data)
vocab = vectorizer.vocabulary_
matrix = []
for i in range(len(vocab)):
  matrix.append([0] * len(vocab))

for i in range(len(vocab)):
  matrix[i][i] = 1

for line in data:
  l = line.split()
  for i in range(1, len(l)):
    if (l[i - 1] in vocab and l[i] in vocab):
      matrix[vocab[l[i - 1]]][vocab[l[i]]] += 1

co_embedding = []

for line in data:
  tmp = [0] * len(vocab)
  for word in line.split():
    if word in vocab:
      tmp[vocab[word]] = sum(matrix[vocab[word]][i] for i in range(0, len(vocab)))
  co_embedding.append(tmp)

#Word2Vec
words = []
for line in data:
  words.append(line.split())

model = Word2Vec(sentences=words,
                 vector_size=100,
                 window=5,
                 min_count=1,
                 workers=4)

wv_embedding = []

for line in data:
  tmp = []
  for word in line.split():
    tmp.append(model.wv[word])
  wv_embedding.append([sum(sub)/len(sub) for sub in zip(*tmp)])
  

"""
Jaccard
"""
matrix = jaccard(data)
heatmap(matrix, 'jaccard')

"""
Manhattan
"""
#One Hot Encoding
matrix = minkowski(oh_embedding, 1)
heatmap(matrix, 'manhattan_oneHot')

#Count Vector
matrix = minkowski(cv_embedding, 1)
heatmap(matrix, 'manhattan_countVector')

#TF-IDF
matrix = minkowski(tf_embedding, 1)
heatmap(matrix, 'manhattan_tfIdf')

#n-grams (2-grams)
matrix = minkowski(ng_embedding, 1)
heatmap(matrix, 'manhattan_nGrams2')

#Co-occurrence Vectors (Context Window = 1)
matrix = minkowski(co_embedding, 1)
heatmap(matrix, 'manhattan_coOccorrence')

#Word2Vec
matrix = minkowski(wv_embedding, 1)
heatmap(matrix, 'manhattan_word2vec')

"""
Euclidean
"""
#One Hot Encoding
matrix = minkowski(oh_embedding, 2)
heatmap(matrix, 'euclidean_oneHot')

#Count Vector
matrix = minkowski(cv_embedding, 2)
heatmap(matrix, 'euclidean_countVector')

#TF-IDF
matrix = minkowski(tf_embedding, 2)
heatmap(matrix, 'euclidean_tfIdf')

#n-grams (2-grams)
matrix = minkowski(ng_embedding, 2)
heatmap(matrix, 'euclidean_nGrams2')

#Co-occurrence Vectors (Context Window = 1)
matrix = minkowski(co_embedding, 2)
heatmap(matrix, 'euclidean_coOccorrence')

#Word2Vec
matrix = minkowski(wv_embedding, 2)
heatmap(matrix, 'euclidean_word2vec')

"""
Minkowski com p=3
"""
#One Hot Encoding
matrix = minkowski(oh_embedding, 3)
heatmap(matrix, 'minkowskiP3_oneHot')

#Count Vector
matrix = minkowski(cv_embedding, 3)
heatmap(matrix, 'minkowskiP3_countVector')

#TF-IDF
matrix = minkowski(tf_embedding, 3)
heatmap(matrix, 'minkowskiP3_tfIdf')

#n-grams (2-grams)
matrix = minkowski(ng_embedding, 3)
heatmap(matrix, 'minkowskiP3_nGrams2')

#Co-occurrence Vectors (Context Window = 1)
matrix = minkowski(co_embedding, 3)
heatmap(matrix, 'minkowskiP3_coOccorrence')

#Word2Vec
matrix = minkowski(wv_embedding, 3)
heatmap(matrix, 'minkowskiP3_word2vec')

"""
Cosine Similarity
"""
#One Hot Encoding
matrix = cosine_similarity(oh_embedding)
heatmap(matrix, 'cosineSimilarity_oneHot')

#Count Vector
matrix = cosine_similarity(cv_embedding)
heatmap(matrix, 'cosineSimilarity_countVector')

#TF-IDF
matrix = cosine_similarity(tf_embedding)
heatmap(matrix, 'cosineSimilarity_tfIdf')

#n-grams (2-grams)
matrix = cosine_similarity(ng_embedding)
heatmap(matrix, 'cosineSimilarity_nGrams2')

#Co-occurrence Vectors (Context Window = 1)
matrix = cosine_similarity(co_embedding)
heatmap(matrix, 'cosineSimilarity_coOccorrence')

#Word2Vec
matrix = cosine_similarity(wv_embedding)
heatmap(matrix, 'cosineSimilarity_word2vec')