# TF-IDF(Term Frequency - Inverse Documentory Frqeuency)

TF-IDF stands for "term frequency-inverse document frequency," a statistical measure used to evaluate the relevance of a term in a document or corpus.

TF-IDF is a widely used method in information retrieval and text mining, and it calculates the importance of a word in a document by taking into account both the frequency of the word in the document and the frequency of the word in the entire corpus of documents.

The "term frequency" component of TF-IDF refers to the number of times a word appears in a particular document. The "inverse document frequency" component refers to how rare or common a word is across all documents in the corpus. This component helps to down-weight the importance of words that appear frequently across all documents in the corpus and boost the importance of words that appear less frequently.

The TF-IDF score of a term in a document is the product of its term frequency and inverse document frequency. A higher TF-IDF score indicates that a word is more important or relevant to the document.

TF-IDF is commonly used in information retrieval tasks, such as search engines, document clustering, and text classification, to identify the most relevant documents for a given query.

****

**Note** 
* This notebook is higly inspired by 
* * **[AskPython](https://www.askpython.com/)=>[Python](https://www.askpython.com/python)=>[Examples](https://www.askpython.com/python/examples)=>[Creating a TF-IDF Model from Scratch](https://www.askpython.com/python/examples/tf-idf-model-from-scratch)**

* * **[YouTube](https://www.youtube.com/)=>[@ritvikmath](https://www.youtube.com/@ritvikmath)=>[TFIDF : Data Science Concepts](https://www.youtube.com/watch?v=OymqCnh-APA)**

****

$$TF = \frac {Number_-of_-times_-a_-word_-"X"_-apprears_-in_-a_-Document}{Number_-Of_-Words_-present_-in_-the_-Document} = \frac {X_n}{n}$$

$$IDF = log(\frac{Number_-of_-Documents_-present_-in_-a_-Corpus}{Number_-of_-Documents_-where_-word_-"X"_-has_-appeared}) = log\frac{X_{n_{unique}}}{D_n}$$
****
$$TF-IDF = \frac {Number_-of_-times_-a_-word_-"X"_-apprears_-in_-a_-Document}{Number_-Of_-Words_-present_-in_-the_-Document} X log(\frac{Number_-of_-Documents_-present_-in_-a_-Corpus}{Number_-of_-Documents_-where_-word_-"X"_-has_-appeared}) = \frac {X_n}{n}log\frac{X_{n_{unique}}}{D_n}$$
