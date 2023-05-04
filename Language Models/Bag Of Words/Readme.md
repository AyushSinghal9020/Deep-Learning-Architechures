# Bag Of Words

Bag of Words (BoW) is a language modeling technique used in natural language processing (NLP) to represent a text document as a bag of its words, disregarding grammar and word order, but keeping track of their frequency.

The basic idea behind BoW is to represent a document as a set of its constituent words, and count how many times each word appears in the document. This results in a sparse vector representation of the document, where the vector is as long as the vocabulary size, and each dimension corresponds to a unique word in the vocabulary.

The BoW technique is often used as a feature extraction method in various NLP tasks, such as text classification, sentiment analysis, and information retrieval. However, it suffers from some limitations, such as not capturing the semantic relationships between words and the context in which they appear. This has led to the development of more advanced techniques, such as word embeddings and deep learning models, that attempt to overcome these limitations.

**Note**
* This notebook is higly inspired by 
* * [Bag Of Word: Natural Language Processing](https://youtu.be/irzVuSO8o4g)
* * [Creating Bag Of Words From S](https://www.askpython.com/python/examples/bag-of-words-model-from-scratch)
