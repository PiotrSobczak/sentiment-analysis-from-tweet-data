import pickle
import numpy as np


class Word2VecMini:
    is_init=False
    word_to_index = None
    embedding_array = None

    @classmethod
    def init(cls):
        if not cls.is_init:
            cls.embedding_array = np.load("data/embeddings_array.numpy")
            cls.word_to_index = pickle.load(open("data/word_to_index.pickle", "rb"))
            cls.is_init = True
            print("Initialized Word2VecMini")
        else:
            print("Word2VecMini already initialized")

    @classmethod
    def get_embedding(cls, word):
        if cls.is_init:
            if word in cls.word_to_index:
                return cls.embedding_array[cls.word_to_index[word]]
            else:
                return np.zeros((1, 400))
        else:
            cls.init()
            return cls.get_embedding(word)
