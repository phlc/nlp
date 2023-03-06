from sklearn.datasets import fetch_20newsgroups as fetch_data
from unidecode import unidecode
import re
from datetime import datetime

"""
Functions
"""

def remove_punctuation(text):
  return re.sub(r"[^\w\s\d]", '', text)



"""
Data processing
"""
data = fetch_data().data
print(data[0])
data = []

for input in fetch_data().data:
  input = unidecode(input)
  input = input.lower()
  input = remove_punctuation(input)
  data.append(input)

print(data[0])

"""
One-hot Encoding
"""


"""
Count Vectors
"""



"""
TF-IDF
"""


"""
N-grams (2-grams)
"""


"""
Co-occurrence Vectors (Context Window = 1)
"""


"""
Word2Vec
"""
