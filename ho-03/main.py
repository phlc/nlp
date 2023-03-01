from unidecode import unidecode
import re
import nltk
from datetime import datetime
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
import csv

nltk.download('stopwords')
nltk.download('wordnet')

"""
Functions
"""


def normalize_dates(text):
  # Regular expression pattern to match date strings in various formats
  date_pattern = r'(\d{1,2}[a-z]{2}\s+of\s+\d{4})|(\d{1,2}[\/\.\-]\d{1,2}[\/\.\-]\d{2,4})|(\d{1,2}[\/\.\-]\d{1,2}[\/\.\-]\d{4})'
  matches = re.findall(date_pattern, text)

  # Loop through all matches and normalize to the format DD/MM/YYYY
  for match in matches:
    original_date = ''.join(match)
    try:
      normalized_date = datetime.strptime(original_date,
                                          '%d%b %Y').strftime('%d/%m/%Y')
    except ValueError:
      try:
        normalized_date = datetime.strptime(original_date,
                                            '%d/%m/%Y').strftime('%d/%m/%Y')
      except ValueError:
        normalized_date = datetime.strptime(original_date,
                                            '%m/%d/%Y').strftime('%d/%m/%Y')
    text = text.replace(original_date, normalized_date)

  return text


def normalize_currency(text):
  # Regular expression pattern to match currency strings in various formats
  currency_pattern = r'\$\s*\d{1,3}(?:[,\.]\d{3})*(?:\.\d{2})?'
  matches = re.findall(currency_pattern, text)

  # Loop through all matches and normalize to the format $DD,DDD.DD
  for match in matches:
    original_currency = match
    amount = original_currency.replace('$', '').replace(',', '').strip()
    normalized_currency = '${:,.2f}'.format(float(amount))
    text = text.replace(original_currency, normalized_currency)

  return text


def remove_punctuation(text):
  # regular expression pattern to match all punctuation except in dates and currencies
  punctuation_pattern = r"[^\w\s\d\/.,$]|(?<!\d)\.|(?<!\d)\/|$(?<!\d)|(?<!\d)\,"

  cleaned_text = re.sub(punctuation_pattern, '', text)

  return cleaned_text


"""
Inputs
"""

#Test Text
test = "It's true, Ms. Martha Töpfer! $3.00 on 3/21/2023 in cash for an ice-cream in the U.S. market? :-( #Truth"

#Shakespeare file
input = open("../datasets/Shakespeare.txt")
shakespeare = input.read()
"""
Text Normalization
"""

testNormalized = unidecode(test)
testNormalized = testNormalized.lower()
testNormalized = normalize_dates(testNormalized)
testNormalized = normalize_currency(testNormalized)
testNormalized = remove_punctuation(testNormalized)

shakespeareNormalized = unidecode(shakespeare)
shakespeareNormalized = shakespeareNormalized.lower()
shakespeareNormalized = normalize_dates(shakespeareNormalized)
shakespeareNormalized = normalize_currency(shakespeareNormalized)
shakespeareNormalized = remove_punctuation(shakespeareNormalized)

print("\nTest String Normalized: " + testNormalized)
output = open("ShakespeareNormalized.txt", "w+")
output.write(shakespeareNormalized)

"""
Text Tokenization
"""
#White Space Tokenization

testToken01 = testNormalized.split()

shakespeareToken01 = shakespeareNormalized.split()

print("\nTest String 1: ", testToken01)
output = open("ShakespeareTokenized01.txt", "w+")
output.write("["+ (", ".join(shakespeareToken01)) +"]")

"""
Stop-words Removal
"""
testToken01_SW = [word for word in testToken01 if not word in stopwords.words()]

sw = stopwords.words('english')

shakespeareToken01_SW = [word for word in shakespeareToken01 if not word in sw]

print("\nTest SW: ", testToken01_SW)
output = open("ShakespeareTokenized01_SW.txt", "w+")
output.write("["+ (", ".join(shakespeareToken01_SW)) +"]")

"""
Text Lemmatization
"""
lemmatizer = WordNetLemmatizer()

testToken01_00 = [lemmatizer.lemmatize(word) for word in testToken01_SW]
ShakespeareTokenized01_00 = [lemmatizer.lemmatize(word) for word in shakespeareToken01_SW]

print("\nTest Lemmatized: ", testToken01_00)
output = open("ShakespeareTokenized01_00.txt", "w+")
output.write("["+ (", ".join(ShakespeareTokenized01_00)) +"]")


"""
Text Stemming
"""
# Porter Stemmer
porter = PorterStemmer()

testToken01_01 = [porter.stem(word) for word in testToken01_SW]
ShakespeareTokenized01_01 = [porter.stem(word) for word in shakespeareToken01_SW]

print("\nTest Porter Stemmer: ", testToken01_01)
output = open("ShakespeareTokenized01_01.txt", "w+")
output.write("["+ (", ".join(ShakespeareTokenized01_01)) +"]")

# Snowball Stemmer
snow = SnowballStemmer("english")

testToken01_02 = [porter.stem(word) for word in testToken01_SW]
ShakespeareTokenized01_02 = [porter.stem(word) for word in shakespeareToken01_SW]

print("\nTest Snowball Stemmer: ", testToken01_02)
output = open("ShakespeareTokenized01_02.txt", "w+")
output.write("["+ (", ".join(ShakespeareTokenized01_02)) +"]")

"""
Vocabulary Analysis
"""

lemmaData = [["Token", "Occurrences", "Size"]]
porterData = [["Token", "Occurrences", "Size"]]
snowData = [["Token", "Occurrences", "Size"]]

for token in ShakespeareTokenized01_00:
  for row in lemmaData:
    if token in row:
      row[1] += 1
    else:
      lemmaData.append([token, 1, len(token ])
  


lemmaFile = csv.writer(open ("ShakespeareTokenized01_00.csv", "w+"))
# porterFile = csv.writer(open ("ShakespeareTokenized01_01.csv", "w+"))
# snowFile = csv.writer(open ("ShakespeareTokenized01_02.csv", "w+"))

