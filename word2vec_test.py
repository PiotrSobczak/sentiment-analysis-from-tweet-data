import os
import pytest

from word2vec import Word2VecWrapper


def test_word2vec_load():
    assert not os.path.isfile("foo.model")
    w2v = Word2VecWrapper()
    with pytest.raises(Exception) as ex:
        w2v.load("foo.model")


def test_word2vec_save():
    MODEL_PATH="foo.model"
    assert not os.path.isfile(MODEL_PATH)
    w2v = Word2VecWrapper()
    w2v.save(MODEL_PATH)
    assert os.path.isfile(MODEL_PATH)
    os.remove(MODEL_PATH)


def test_word2vec_train():
    w2v = Word2VecWrapper()
    with pytest.raises(Exception) as ex:
        w2v.model.wv["hello"]
    w2v.train("/home/piotrsobczak/magisterka-dane/crowdflower-tweets/text_emotion.csv")
    w2v.model.wv["hello"]
