## What is the purpose of this project?
The goal of this project is to create system for opinion mining using Deep Learning on sentiment140 dataset.

## Hardware
The training was conducted using GeForce GTX 1060 graphic card

## Data

<img src="https://github.com/PiotrSobczak/emotion-recognition-from-speech/blob/master/assets/data.png" width="600"></img>

For the purpose of the project, [sentiment140](http://help.sentiment140.com/for-students) dataset was used due to it’s size and availability.
The dataset was created by Stanford University during reasearch on sentiment analysis on
twitter data, more can be found in this paper. The data consists of 1.6 million tweets annotated
with 2 classes.(positive or negative sentiment). They gathered the data using Tweeter API.

## Preprocessing  
The next step is data preprocessing. I made the following preprocessing operations:
- Convert text to lowercase
- Removing tags (@mike)
- Removing links (https:..)
- Removing numbers
- Removing special signs(“(“, “)“, “-“, “/“)
- Remove words which are not in out Word2Vec(Skip-Gram) vocabulary.
- Remove tweets which exceed empirically chosen sequence length.

## Word Embeddings   
In order to train a high quality system having limited resources, I’ve chosen to use a pretrained Word2Vec SkipGram model to obtain word representations in for of distributed vectors called embeddings. The Skip-Gram model was trained on 400 millions of tweets, has a vocabulary of over 3 milions words, and embedding dimension of 400. More about the model architecture can be read in this [paper](https://mail.google.com/mail/u/0/#search/drasza/QgrcJHsNnjWZGsJFSqZKklBHfqsJdSmMLnv?projector=1&messagePartId=0.1). Using this model we can obtain very meaningful word representations for our classifier, which allowed me to use a pretty shallow LSTM network and achieve results of 85% accuracy after training for approximately 1 hour. This wouldn’t be possible if I used for example one-hot encoding.

## Classifier Architecture  
<img src="https://github.com/PiotrSobczak/emotion-recognition-from-speech/blob/master/assets/architecture.png" width="600"></img>

Parameters of classifier:
- sequence length - chosen based on data as 30. (Only about 1.5 % of tweets had more
than 30 words but about 20% had more than 20 words)
- batch size - was chosen empirically as 64
- embedding size - defined by Skip-Gram model(400)

## Training
In order to find suboptimal parameters for my network i conducted a hyper-parameter tuning
on 3 different machines during which I am randomly choosing a set of hyperparameters from
given range.
Tuned hyperparameters were:
- Hidden dim of lstm
- Number of layers of lstm
- Dropout ratio in 3 places(on hidden state of last lstm layer,between lstm layers and on
input_embeddings)
- L2 regularization ratio
- Bidirectionality

The best achieved result was 85.21% accuracy on validation set and 84.53% accuracy on test
set.
- Hidden dim of lstm = 850
- Number of layers of lstm = 1
- Dropout ratio = 0.6
- L2 regularization ratio = 1e-5
- Bidirectionality = Yes
Comparable results but were achieved using no-bidirectional, but with two-layer lstm.

## Sentiment Classifier Demo   
```diff
+ [yes] 0.466 sentiment :)  
+ [yes ! ! !] 0.989 sentiment :)  
- [no] -0.987 sentiment :(  
- [no ! ! !] -0.925 sentiment :(  
+ [i like cars deadly] 0.992 sentiment :)  
- [this car is like deadly] -0.96 sentiment :(  
+ [feeling well] 0.569 sentiment :)  
- [not feeling well] -1.0 sentiment :(  
+ [not , i'm feeling well] 0.973 sentiment :)  
+ [he is cool] 1.0 sentiment :)  
+ [he thinks he is cool] 0.999 sentiment :)  
- [he thinks he is cool , but he is not] -0.958 sentiment :(  
- [a puppy that was born recently was put to sleep] -0.086 sentiment :(  
- [a puppy was put to sleep] -0.483 sentiment :(  
- [a puppy that was born recently was euthanized] -0.931 sentiment :(  
- [a puppy that was born happy recently was euthanized] -0.707 sentiment :(  
+ [put to sleep] 0.689 sentiment :)  
- [i put my puppy to sleep] -0.47 sentiment :(  
+ [i put my daughter to sleep] 0.093 sentiment :)  
```

## Results
My system achieved the following results on sentiment140 dataset.
Accuracy on validation set: 85,21[%]
Accuracy on test set: 84,53[%]


<img src="https://github.com/PiotrSobczak/emotion-recognition-from-speech/blob/master/assets/results.png" width="600"></img>

References:
[[1](https://aaai.org/ocs/index.php/FLAIRS/FLAIRS17/paper/download/15430/14952)]
[[2](https://books.google.pl/books?id=WQRrDwAAQBAJ&pg=PA234&lpg=PA234&dq=sentiment140+f1+score&source=bl&ots=kFYaA7Qz3r&sig=ACfU3U3CEAKCviWY72GNDrEq6ixffYTzmQ&hl=pl&sa=X&ved=2ahUKEwj_0bvcxofgAhWQKCwKHXFMCGkQ6AEwAHoECAkQAQ#v=onepage&q=sentiment140%20%20f1%20score&f=false)]
[[3](https://cora.ucc.ie/bitstream/handle/10468/3972/2619.pdf?sequence=1)]
[[4](https://www.sciencedirect.com/science/article/pii/S0957417417300751)]
[[5](https://arxiv.org/pdf/1806.02863.pdf)]
