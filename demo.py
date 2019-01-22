from preprocessing import Preprocessor
from word2vec_wrapper import Word2VecWrapper
from train import RNN
import torch
import numpy as np
import json


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
    model = load_model("models/2019-01-21_17:32:30_84acc/model")

    with torch.no_grad():
        sentiment = model(text_embedded)
        print("Sentence {} has {} sentiment".format(text, float(sentiment)))


if __name__=="__main__":
    demo("yes")
    import pdb;pdb.set_trace()