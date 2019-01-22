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
    model = load_model("models/2019-01-21_17:32:30_84acc/model")

    with torch.no_grad():
        sentiment = model(text_embedded)
        print("[{}] {} sentiment".format(text, float(tanh(sentiment))))


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
    import pdb;pdb.set_trace()