from unidecode import unidecode
import nltk
import nltk.tokenize as to
import textblob as tb
import regex as re

nltk.download('punkt')

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
testNormalized = re.sub('[^\w\d\s$/]', '', testNormalized)


shakespeareNormalized = unidecode(shakespeare)
shakespeareNormalized = shakespeareNormalized.lower()
shakespeareNormalized = re.sub('[^\w\d\s$/]', '', shakespeareNormalized)

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

# #NLTK: Word Tokenizer

testToken02 = to.word_tokenize(testNormalized)

shakespeareToken02 = to.word_tokenize(shakespeareNormalized)

print("\nTest String 2: ", testToken02)
output = open("ShakespeareTokenized02.txt", "w+")
output.write("["+ (", ".join(shakespeareToken02)) +"]")

# #NLTK: Tree Bank Tokenizer

testToken03 = to.TreebankWordTokenizer().tokenize(testNormalized)

shakespeareToken03 = to.TreebankWordTokenizer().tokenize(shakespeareNormalized)

print("\nTest String 3: ", testToken03)
output = open("ShakespeareTokenized03.txt", "w+")
output.write("["+ (", ".join(shakespeareToken03)) +"]")

# #NLTK: Word Punctuation Tokenizer

testToken04 = to.WordPunctTokenizer().tokenize(testNormalized)

shakespeareToken04 = to.WordPunctTokenizer().tokenize(shakespeareNormalized)

print("\nTest String 4: ", testToken04)
output = open("ShakespeareTokenized04.txt", "w+")
output.write("["+ (", ".join(shakespeareToken04)) +"]")

# #NLTK: Tweet Tokenizer

testToken05 = to.TweetTokenizer().tokenize(testNormalized)

shakespeareToken05 = to.TweetTokenizer().tokenize(shakespeareNormalized)

print("\nTest String 5: ", testToken05)
output = open("ShakespeareTokenized05.txt", "w+")
output.write("["+ (", ".join(shakespeareToken05)) +"]")

# #NLTK: MWE Tokenizer

tk = to.MWETokenizer([('$', '300')], separator='')

testToken06 = tk.tokenize(testToken05)

shakespeareToken06 = tk.tokenize(shakespeareToken05)

print("\nTest String 6: ", testToken06)
output = open("ShakespeareTokenized06.txt", "w+")
output.write("["+ (", ".join(shakespeareToken06)) +"]")

# #TextBlob Word Tokenizer

testToken07 = tb.TextBlob(testNormalized).words

shakespeareToken07 = tb.TextBlob(shakespeareNormalized).words

print("\nTest String 7: ", testToken07)
output = open("ShakespeareTokenized07.txt", "w+")
output.write("["+ (", ".join(shakespeareToken05)) +"]")

# #spaCy Tokenizer

# print("Test String 8: " + testToken08)
# output = open("ShakespeareTokenized08.txt", "w+")
# output.write(shakespeareToken08)

# #Gensim Word Tokenizer

# print("Test String 9: " + testToken09)
# output = open("ShakespeareTokenized09.txt", "w+")
# output.write(shakespeareToken09)

# #Keras Tokenization

# print("Test String 10: " + testToken10)
# output = open("ShakespeareTokenized10.txt", "w+")
# output.write(shakespeareToken10)
