# Basic Concepts

## Text
Text is anything that can be read. It is a symbolic arrangement of letters in a piece of writing, a coherent set of signs that transmits some kind of informative message.

```
"After you have chosen your words, they must be weaved together into a fine and delicate fabric."
Institutio Oratoria, Quintilian (Roman Educator), 95 BC
```

> From Latin: **Textum** = **Fabric**


## Text Mining
Deriving high-quality information from text
  - High-quality -> relevance, novelty or interest
  - Discovery by computer of new, previously unknown information, by automatically extracting information from different written resources (text)
  - Structuring the input text -> Derive patterns within the structured data -> Evaluation and interpretation of the output
  - Turn text into data for analysis, via application of natural language processing (NLP) and different types of analytical methods

> Synonyms: Text Data Mining (TDM), Text Analytics
>
> Related Terms: Data Mining, Knowledge Discovery




## Text Normalization
Transforming text into a single canonical form.
> Synonyms: Canonicalization
>
> Related Terms: Morphology Transformation

Challenges:
  - anti-discriminatory, antidiscriminatory -> antidiscriminatory
  - pêra, pera -> pera
  - black, Black -> black
  - CAT, cat -> cat
  - colour, color -> color
  - 2/3/2021 -> Mar, 2 2021
  - 230.423 203,423 -> 230423
  - Rindfleischetikettierungsueberwachungsaufgabenuebertragungsgesetz -> lei que delega a monitorização da carne de vaca

It frequently evolves case-transformation, punctuation, accents and special characters removal

Morfologia -> Composição textual   window, window + ing (ação), window + s (plural)

Sintaxe -> Composição gramatical do texto    Substantivo? Verbo? Adjetivo?

Semântica -> Significado dos componentes textuais -> conceito, ideia

## Text Segmentation
Non-trivial problem of dividing a string of written language into its components.

  - Morphological Segmentation: String -> Morphemes
  - Word Segmentation: String -> words
  - Intent Segmentation: String -> 2 or more group of words
  - Sentence Segmentation: String -> sentences
  - Paragraph Segmentation: String -> paragraphs
  - Topic Segmentation: String -> topics (groups of sentences on the same topic)

> See in [Wikipedia](https://en.wikipedia.org/wiki/Text_segmentation#:~:text=Text%20segmentation%20is%20the%20process,subject%20of%20natural%20language%20processing).

## Text Tokenization
It's a [Text Segmentation](#text-segmentation) problem of breaking the raw text into small chunks, such as words, terms, sentences, symbols, or some other meaningful elements called tokens.

A tokenizer breaks unstructured data and natural language text into chunks of discrete elements, where token occurrences can be used as a vector representing text. This turns text into a numerical data structure suitable for NLP and machine learning.


Challenges:
  - The boundary of a word varies in different languages;
  - Form variations (It is and It's)

Approaches:
  - White Space Tokenization
  - Rule Based Tokenization (Dictionary Based Tokenization, Regular Expression Tokenization)
  - Penn TreeBank Tokenization
  - Spacy Tokenizer
  - Moses Tokenizer
  - Subword Tokenization (BPE, WordPiece, Unigram Language Model)

Python Libraries:
  - NLTK
  - TextBlob
  - spaCy
  - Gensim
  - Keras

> See in Book [Chapter](https://nlp.stanford.edu/IR-book/html/htmledition/tokenization-1.html).
