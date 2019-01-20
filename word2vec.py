from gensim.models.word2vec import Word2Vec
from gensim.utils import simple_preprocess
import logging
from data_loader import DataLoader
from nltk.tokenize import PunktSentenceTokenizer
import multiprocessing as mp

logging.basicConfig(level=logging.INFO)


class Word2VecWrapper:
    def __init__(self):
        self.model = Word2Vec()

    def train(self, dataset_path="/home/piotrsobczak/Downloads/magisterka-dane/crowdflower-tweets/text_emotion.csv"):
        # Loading dataset
        dataset = DataLoader.load_crowdflower_db(dataset_path)

        # Spliting tweets into sentences and words
        all_tweets = sum(dataset.values(), [])
        tok = PunktSentenceTokenizer()
        all_tweets_to_sentences = sum([tok.tokenize(tweet) for tweet in all_tweets], [])
        all_tweets_to_words = [simple_preprocess(sentence) for sentence in all_tweets_to_sentences]

        # Building Word2Vec vocab model and training
        self.model = Word2Vec(
            sentences=all_tweets_to_words,
            size=512,
            window=10,
            negative=20,
            iter=50,
            seed=1000,
            workers=mp.cpu_count()
        )

        self.model.train(all_tweets_to_words, total_examples=self.model.corpus_count, epochs=50)

    def load(self, load_path="word2vec.model"):
        self.model = Word2Vec.load(load_path)

    def save(self, save_path="word2vec.model"):
        self.model.save(save_path)