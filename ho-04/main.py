from sklearn.datasets import fetch_20newsgroups as fetch_data
from unidecode import unidecode
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
"""
Functions
"""


# Punctuation Removal
def remove_punctuation(text):
  return re.sub(r"[^\w\s\d]", '', text)


# One Hot Encoding
def one_hot_encoding(input):
  tokens = []
  for word in input.split():
    if word not in tokens:
      tokens.append(word)
  tokens.sort()

  final_encoding = []

  for line in input.split('\n'):
    line_encoding = []
    line_dict = dict.fromkeys(tokens, 0)
    for word in line.split():
      line_dict[word] = 1
    for word in tokens:
      line_encoding.append(line_dict[word])
    final_encoding.append(line_encoding)

  return final_encoding


# Count Vectors
def count_vectors(input):
  tokens = []
  for word in input.split():
    if word not in tokens:
      tokens.append(word)
  tokens.sort()

  final_encoding = []

  for line in input.split('\n'):
    line_encoding = []
    line_dict = dict.fromkeys(tokens, 0)
    for word in line.split():
      line_dict[word] += 1
    for word in tokens:
      line_encoding.append(line_dict[word])
    final_encoding.append(line_encoding)

  return final_encoding


# Two Grams Count Vectors
def two_grams_count_vectors(input):
  two_grams_set = []
  tokens = []
  for line in input.split('\n'):
    two_grams_line = []
    words = line.split()
    for i in range(1, len(words)):
      tokens.append(f'{words[i-1]} {words[i]}')
      two_grams_line.append(f'{words[i-1]} {words[i]}')
    two_grams_set.append(two_grams_line)

  tokens.sort()

  final_encoding = []

  for line in two_grams_set:
    line_encoding = []
    line_dict = dict.fromkeys(tokens, 0)
    for word in line:
      line_dict[word] += 1
    for word in tokens:
      line_encoding.append(line_dict[word])
    final_encoding.append(line_encoding)

  return final_encoding


# Co-Occurrence Matrix - Window 1
def co_occurrence(input):
  tokens = {word for word in input.split()}
  tokens = {token: i for token, i in zip(tokens, range(len(tokens)))}

  matrix = []
  for i in range(0, len(tokens)):
    matrix.append([0] * len(tokens))

  for line in input.split('\n'):
    line = line.split()
    for i in range(1, len(line)):
      matrix[tokens[line[i - 1]]][tokens[line[i]]] += 1

  return matrix


"""
Data processing
"""
data = []

for input in fetch_data().data:
  input = unidecode(input)
  input = input.lower()
  input = remove_punctuation(input)
  data.append(input)

data = data[0:10]

"""
One-hot Encoding
"""
with open("20News_01.txt", "w+") as f:
  for input in data:
    input_encoding = one_hot_encoding(input)
    for line in input_encoding:
      line = [str(i) for i in line]
      f.write("[" + ", ".join(line) + "]\n")
    f.write('\nNew input\n')

"""
Count Vectors
"""
with open("20News_02.txt", "w+") as f:
  for input in data:
    input_encoding = count_vectors(input)
    for line in input_encoding:
      line = [str(i) for i in line]
      f.write("[" + ", ".join(line) + "]\n")
    f.write('\nNew input\n')
    
"""
TF-IDF
"""
tf_idf = TfidfVectorizer(smooth_idf=False)
with open("20News_03.txt", "w+") as f:
  for input in data:
    input_encoding = tf_idf.fit_transform(input.split('\n')).toarray().tolist()
    for line in input_encoding:
      line = [str(i) for i in line]
      f.write("[" + ", ".join(line) + "]\n")
    f.write('\nNew input\n')
    
"""
N-grams (2-grams)
"""
with open("20News_04.txt", "w+") as f:
  for input in data:
    input_encoding = two_grams_count_vectors(input)
    for line in input_encoding:
      line = [str(i) for i in line]
      f.write("[" + ", ".join(line) + "]\n")
    f.write('\nNew input\n')
    
"""
Co-occurrence Vectors (Context Window = 1)
"""
with open("20News_05.txt", "w+") as f:
  for input in data:
    input_encoding = co_occurrence(input)
    for line in input_encoding:
      line = [str(i) for i in line]
      f.write("[" + ", ".join(line) + "]\n")
    f.write('\nNew input\n')
    
"""
Word2Vec
"""
with open("20News_06.txt", "w+") as f:
  for input in data:
    words = []
    for line in input.split('\n'):
      words.append(line.split())
    model = Word2Vec(sentences=words,
                     vector_size=100,
                     window=5,
                     min_count=1,
                     workers=4)

    for line in model.wv.vectors:
      line = [str(i) for i in line]
      f.write("[" + ", ".join(line) + "]\n")
    f.write('\nNew input\n')