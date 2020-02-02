from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import numpy as np
import os
import warnings
from string import punctuation
from nltk.corpus import stopwords

def train_word2vec(data_dirs, n_feature, model_path, sg):
    # make dictionary
    print("Making document list...")
    documents = []
    
    # train_documents = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]

    train_documents = []
    for data_dir in data_dirs:
        for f in os.listdir(data_dir):
            train_documents.append(os.path.join(data_dir, f))

    print(train_documents)

    for doc in train_documents:
        doc_content = []
        with open(doc) as d:
            for line in d:
                words = line.split()
                for word in words:
                    doc_content.append(word)
            documents.append(doc_content)

    print("Training Word2Vec model...")
    model_word2vec = Word2Vec(
        documents, size=n_feature, window=5, min_count=5, sg=sg)
    model_word2vec.train(documents, total_examples=len(documents), epochs=10)
    model_word2vec.save(model_path)
    print("Training complete!!!")


def extract_word2vec(model_path, word, n_feature=100):
    trained_model = Word2Vec.load(model_path)
    # words = list(trained_model.wv.vocab)
    # print words

    # extract train matrix to csv
    #print("Extracting Word2Vec of word "+word)
    try:
        word_vec = trained_model.wv[word].reshape(n_feature)
        return word_vec
    except:
        word_vec = np.zeros((n_feature))
        return word_vec