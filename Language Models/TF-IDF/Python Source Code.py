import numpy as np 
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')

def TF_IDF(data):
    vocab = []

    for _ in range(len(data)):
        k = [j.lower() for j in word_tokenize(data[0]) if j.isalpha()]
        data.append(k)
        data.pop(0)
        for word in k:
            if word not in vocab:
                vocab.append(word)

    indices = {}
    i = 0
    for word in vocab:
        indices[word] = i
        i += 1

    def count_dict(data):
        word_count = {}
        for word in vocab:
            word_count[word] = 0
            for sent in data:
                if word in sent:
                    word_count[word] += 1
        return word_count
    word_count = count_dict(data)

    def termfreq(document, word):
        N = len(document)
        occurance = len([token for token in document if token == word])
        return occurance/N

    def inverse_doc_freq(word):
        try:
            word_occurance = word_count[word] + 1
        except:
            word_occurance = 1 
        return np.log(len(data)/word_occurance)

    def tf_idf(sentence):
        tf_idf_vec = np.zeros((len(vocab),))
        for word in sentence:
            tf = termfreq(sentence,word)
            idf = inverse_doc_freq(word)        
            value = tf*idf
            tf_idf_vec[indices[word]] = value 
        return tf_idf_vec

    vectors = []
    for sent in data:
        vec = tf_idf(sent)
        vectors.append(vec)
        
    return vectors
