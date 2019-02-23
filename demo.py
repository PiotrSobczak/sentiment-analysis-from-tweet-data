from preprocessing import Preprocessor
from word2vec_wrapper import Word2VecWrapper
from train import RNN
from torch.nn.functional import tanh
import numpy as np
import json
import torch


def load_model(model_path):
    config = json.load(open("{}.config".format(model_path), "r"))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == "cuda":
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    model = RNN(config=config)
    model = model.to(device)
    model.float()
    model.load_state_dict(torch.load("{}.torch".format(model_path)))
    model.eval()
    return model


def demo(text):
    text = Preprocessor.preprocess_one(text)
    text_embedded = Word2VecWrapper.get_sentence_embedding(text, 30)
    text_embedded = np.expand_dims(text_embedded, 0).swapaxes(0, 1)
    model = load_model("models/2019-01-24_21:25:57_test_loss_84_53_val_loss_85.21/model")

    with torch.no_grad():
        sentiment = model(text_embedded)
        sentiment = float(tanh(sentiment))
        emoticon = ":)" if sentiment > 0 else ":("
        print("[{}] {} sentiment {}".format(text, round(sentiment, 3), emoticon))


if __name__=="__main__":
    demo("yes")
    demo("yes!!!")

    demo("no")
    demo("no!!!")

    demo("i like cars deadly")
    demo("this car is like deadly")

    demo("feeling well")
    demo("not feeling well")
    demo("not, i'm feeling well")

    demo("he is cool")
    demo("he thinks he is cool")
    demo("he thinks he is cool, but he is not")
    demo("he thinks he is cool, but that's a lie")

    demo("A puppy that was born recently was put to sleep")
    demo("A puppy was put to sleep")
    demo("A puppy that was born recently was euthanized")
    demo("A puppy that was born happy recently was euthanized")
    demo("put to sleep")
    demo("i put my puppy to sleep")
    demo("i put my daughter to sleep")
    import pdb;pdb.set_trace()