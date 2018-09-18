from gensim.models.word2vec import Word2Vec
from gensim.utils import simple_preprocess
import logging
from data_loader import DataLoader
from nltk.tokenize import PunktSentenceTokenizer
import multiprocessing as mp

logging.basicConfig(level=logging.INFO)


class Word2VecUtils:
    @staticmethod
    def train(
            dataset_path="/home/piotrsobczak/Downloads/magisterka-dane/crowdflower-tweets/text_emotion.csv",
            save_path="word2vec.model"
    ):
        # Loading dataset
        dataset = DataLoader.load_crowdflower_db(dataset_path)

        # Spliting tweets into sentences and words
        all_tweets = sum(dataset.values(), [])
        tok = PunktSentenceTokenizer()
        all_tweets_to_sentences = sum([tok.tokenize(tweet) for tweet in all_tweets], [])
        all_tweets_to_words = [simple_preprocess(sentence) for sentence in all_tweets_to_sentences]

        # Building Word2Vec vocab model and training
        model = Word2Vec(sentences=all_tweets_to_words,
                         size=512,
                         window=10,
                         negative=20,
                         iter=50,
                         seed=1000,
                         workers=mp.cpu_count())
        model.train(all_tweets_to_words, total_examples=model.corpus_count, epochs=50)
        model.save(save_path)

    @staticmethod
    def load(load_path="word2vec.model"):
        model = Word2Vec.load(load_path)
        return model.wv
