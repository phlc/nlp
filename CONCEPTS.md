# Basic Concepts

## Text
> From Latin: **Textum** = **Fabric**

```
"After you have chosen your words, they must be weaved together into a fine and delicate fabric."
Institutio Oratoria, Quintilian (Roman Educator), 95 BC
```

Text is anything that can be read. It is a symbolic arrangement of letters in a piece of writing, a coherent set of signs that transmits some kind of informative message.

## Text Mining
> Synonyms: Text Data Mining (TDM), Text Analytics
>
> Related Terms: Data Mining, Knowledge Discovery

- Deriving high-quality information from text, where high-quality means relevance, novelty or interest
- Discovery by computer of new, previously unknown information, by automatically extracting information from different written resources (text)
- Structuring the input text -> Derive patterns within the structured data -> Evaluation and interpretation of the output
- Turn text into data for analysis, via application of natural language processing (NLP) and different types of analytical methods


E = M x C2

Energia = Massa x Velocidade da Luz

= é conversível

x é associação linear

C2 é constante

Dados -> Símbolos, sinais, códigos

Informação -> Conceitos, ideias

Conhecimento -> Correlação

## Text Normalization (Transformação morfológica)
> Synonyms: Canonicalization
>
> Related Terms: Morphology Transformation

Text Normalization consists in transforming text into a single canonical form.

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
It is a non-trivial problem of dividing a string of written language into its components.

Morphological Segmentation: String -> Morphemes
Word Segmentation: String -> words
Intent Segmentation: String -> 2 or more group of words
Sentence Segmentation: String -> sentences
Paragraph Segmentation: String -> paragraphs
Topic Segmentation: String -> topics (groups of sentences on the same topic)

https://en.wikipedia.org/wiki/Text_segmentation#:~:text=Text%20segmentation%20is%20the%20process,subject%20of%20natural%20language%20processing.

## Text Tokenization
It's a Text Segmentation Problem.

Breaking the raw text into small chunks, such as words, terms, sentences, symbols, or some other meaningful elements called tokens.

A tokenizer breaks unstructured data and natural language text into chunks of discrete elements, where token occurrences can be used as a vector representing text. This turns text into a numerical data structure suitable for NLP and machine learning.

https://nlp.stanford.edu/IR-book/html/htmledition/tokenization-1.html

Challenge: The boundary of a word varies in different languages, as well as form variations (It is and It's)
Delimiters of Words: whitespace character, punctuation characters.

White Space Tokenization, Rule Based Tokenization (Dictionary Based Tokenization, Regular Expression Tokenization), Penn TreeBank Tokenization, Spacy Tokenizer, Moses Tokenizer, Subword Tokenization (BPE, WordPiece, Unigram Language Model)

Python Libraries: NLTK, TextBlob, spaCy, Gensim, Keras

NLTK Rule-based: Treebank, Tweet, MWET (Multi-Word Expression Tokenizer)
