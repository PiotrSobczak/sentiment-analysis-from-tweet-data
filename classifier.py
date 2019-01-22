import torch.nn as nn
import torch
import torch.optim as optim
from data_loader import BatchLoader, DataLoader
import torch.tensor
from utils import timeit
import json
import os
from time import gmtime, strftime

EMBEDDING_DIM = 400
HIDDEN_DIM = 800
OUTPUT_DIM = 1
DROPOUT = 0.0
N_EPOCHS = 100
PATIENCE = 3
MODEL_PATH = "models"
MODEL_RUN_PATH = MODEL_PATH + "/" + strftime("%Y-%m-%d_%H:%M:%S", gmtime())
MODEL_CONFIG = "{}/model.config".format(MODEL_RUN_PATH)
MODEL_WEIGHTS = "{}/model.torch".format(MODEL_RUN_PATH)
REG_RATIO=0.00001
NUM_LAYERS=1


class RNN(nn.Module):
    def __init__(self, embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM, dropout=DROPOUT, config=None):
        super().__init__()

        if config is not None:
            embedding_dim = int(config["embedding_dim"])
            hidden_dim = int(config["hidden_dim"])
            dropout = float(config["dropout"])
	    
            print("Loaded custom config")
        else:
            json.dump({"embedding_dim": embedding_dim, "hidden_dim": hidden_dim, "dropout": dropout, "reg_ratio": REG_RATIO, "n_layers": NUM_LAYERS}, 
open(MODEL_CONFIG, 
"w"))
            print("Saved model config to {}".format(MODEL_CONFIG))

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=NUM_LAYERS)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        # x = [sent len, batch size, emb dim]
        x = torch.cuda.FloatTensor(x)
        output, (hidden, cell) = self.lstm(x)

        # output = [sent len, batch size, hid dim * num directions]
        # hidden = [num layers * num directions, batch size, hid dim]
        # cell = [num layers * num directions, batch size, hid dim]

        # concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
        # and apply dropout
        hidden = hidden[-1, :, :]
        #hidden = self.dropout(hidden)
        hidden = hidden.squeeze(0).float()
        return self.fc(hidden)


def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division
    acc = correct.sum()/len(correct)
    return acc


@timeit
def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch, labels in iterator():
        optimizer.zero_grad()

        predictions = model(batch).squeeze(1)

        loss = criterion(predictions, labels)
        acc = binary_accuracy(predictions, labels)

        reg_loss = 0
        for param in model.parameters():
            reg_loss += param.norm(2)

        total_loss = loss + REG_RATIO*reg_loss
        total_loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

@timeit
def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for batch, labels in iterator():
            predictions = model(batch).squeeze(1)

            loss = criterion(predictions, labels)

            acc = binary_accuracy(predictions, labels)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


if __name__ == "__main__":
    os.makedirs(MODEL_RUN_PATH, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == "cuda":
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    model = RNN(EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, DROPOUT)
    model.float()

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    criterion = nn.BCEWithLogitsLoss()
    model = model.to(device)
    criterion = criterion.to(device)

    best_valid_loss = 999
    epochs_without_improvement = 0

    train_raw, val_raw, test_raw = DataLoader.get_data_in_batches()
    train_iterator = BatchLoader(train_raw)
    validation_iterator = BatchLoader(val_raw)
    test_iterator = BatchLoader(test_raw)

    for epoch in range(N_EPOCHS):
        if epochs_without_improvement == PATIENCE:
            break

        valid_loss, valid_acc = evaluate(model, validation_iterator, criterion)
        print(f'| Epoch: {epoch:02} | Val Loss: {valid_loss:.4f} | Val Acc: {valid_acc*100:.3f}%')
        if valid_loss < best_valid_loss:
            torch.save(model.state_dict(), MODEL_WEIGHTS)
            print("Val loss improved from {} to {}. Saving model to {}.".format(best_valid_loss, valid_loss, MODEL_WEIGHTS))
            best_valid_loss = valid_loss
            epochs_without_improvement = 0

        train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
        print(f'| Epoch: {epoch+1:02} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.3f}%')

        epochs_without_improvement += 1

    model.load_state_dict(torch.load(MODEL_WEIGHTS))
    test_loss, test_acc = evaluate(model, test_iterator, criterion)
    print(f'| Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}% |')
