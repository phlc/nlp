from unidecode import unidecode
import re
from datetime import datetime


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
