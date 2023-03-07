from sklearn.datasets import fetch_20newsgroups as fetch_data
from unidecode import unidecode
import re
from sklearn.feature_extraction.text import TfidfVectorizer


"""
Functions
"""
def remove_punctuation(text):
  return re.sub(r"[^\w\s\d]", '', text)


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

# """
# One-hot Encoding
# """

with open("20News_01.txt", "w+") as f:
  for input in data:
    input_encoding = one_hot_encoding(input)
    for line in input_encoding:
      line = [str(i) for i in line]
      f.write("[" + ", ".join(line) + "]\n")
    f.write('\nNew input\n')
    
# """
# Count Vectors
# """

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
"""
Co-occurrence Vectors (Context Window = 1)
"""
"""
Word2Vec
"""
