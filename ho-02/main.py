from unidecode import unidecode
import re
from datetime import datetime
import nltk.tokenize as to
import textblob as tb
import spacy
import gensim.utils as gensim
from keras.preprocessing.text import text_to_word_sequence as keras
import nltk
nltk.download('punkt')

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

#NLTK: Word Tokenizer

testToken02 = to.word_tokenize(testNormalized)

shakespeareToken02 = to.word_tokenize(shakespeareNormalized)

print("\nTest String 2: ", testToken02)
output = open("ShakespeareTokenized02.txt", "w+")
output.write("["+ (", ".join(shakespeareToken02)) +"]")

#NLTK: Tree Bank Tokenizer

testToken03 = to.TreebankWordTokenizer().tokenize(testNormalized)

shakespeareToken03 = to.TreebankWordTokenizer().tokenize(shakespeareNormalized)

print("\nTest String 3: ", testToken03)
output = open("ShakespeareTokenized03.txt", "w+")
output.write("["+ (", ".join(shakespeareToken03)) +"]")

#NLTK: Word Punctuation Tokenizer

testToken04 = to.WordPunctTokenizer().tokenize(testNormalized)

shakespeareToken04 = to.WordPunctTokenizer().tokenize(shakespeareNormalized)

print("\nTest String 4: ", testToken04)
output = open("ShakespeareTokenized04.txt", "w+")
output.write("["+ (", ".join(shakespeareToken04)) +"]")

#NLTK: Tweet Tokenizer

testToken05 = to.TweetTokenizer().tokenize(testNormalized)

shakespeareToken05 = to.TweetTokenizer().tokenize(shakespeareNormalized)

print("\nTest String 5: ", testToken05)
output = open("ShakespeareTokenized05.txt", "w+")
output.write("["+ (", ".join(shakespeareToken05)) +"]")

#NLTK: MWE Tokenizer

tk = to.MWETokenizer([('$', '300')], separator='')

testToken06 = tk.tokenize(testToken05)

shakespeareToken06 = tk.tokenize(shakespeareToken05)

print("\nTest String 6: ", testToken06)
output = open("ShakespeareTokenized06.txt", "w+")
output.write("["+ (", ".join(shakespeareToken06)) +"]")

#TextBlob Word Tokenizer

testToken07 = tb.TextBlob(testNormalized).words

shakespeareToken07 = tb.TextBlob(shakespeareNormalized).words

print("\nTest String 7: ", testToken07)
output = open("ShakespeareTokenized07.txt", "w+")
output.write("["+ (", ".join(shakespeareToken07)) +"]")

#spaCy Tokenizer

sp = spacy.load('en_core_web_sm')

testToken08 = [token.text for token in sp(testNormalized)]

shakespeareToken08 = [token.text for token in sp(shakespeareNormalized)]

print("\nTest String 8: ", testToken08)
output = open("ShakespeareTokenized08.txt", "w+")
output.write("["+ (", ".join(shakespeareToken08)) +"]")

#Gensim Word Tokenizer

testToken09 = list(gensim.tokenize(testNormalized))

shakespeareToken09 = list(gensim.tokenize(shakespeareNormalized))

print("\nTest String 9: ", testToken09)
output = open("ShakespeareTokenized09.txt", "w+")
output.write("["+ (", ".join(shakespeareToken09)) +"]")

#Keras Tokenization

testToken10 = keras(testNormalized)

shakespeareToken10 = keras(shakespeareNormalized)

print("\nTest String 10: ", testToken10)
output = open("ShakespeareTokenized10.txt", "w+")
output.write("["+ (", ".join(shakespeareToken10)) +"]")
