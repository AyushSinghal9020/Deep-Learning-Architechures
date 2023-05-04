import numpy as np 
from collections import defaultdict
import nltk
nltk.download('punkt') 

def BagOfWords(data):
    
    sentences = []
    vocab = []
    
    for sent in data:
    
        sentence = [w.lower() 
                    for w in nltk.tokenize.word_tokenize(sent) 
                    if w.isalpha() ]
        
        sentences.append(sentence)
        
        for word in sentence:
        
            if word not in vocab:
            
                vocab.append(word)
                
    index_word = {}
    
    i = 0
    
    for word in vocab:
    
        index_word[word] = i 
        i += 1
        
    count_dict = defaultdict(int)
    vec = np.zeros(len(vocab))
    
    for item in sent:
    
        count_dict[item] += 1
    
    for key,item in count_dict.items():
        
        vec[index_word[key]] = item
    
    return vec   
