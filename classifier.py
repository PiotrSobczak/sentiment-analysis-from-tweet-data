import torch.nn as nn
import torch
import torch.optim as optim
from data_loader import BatchLoader, DataLoader
import torch.tensor
from utils import timeit


class RNN(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim):
        super().__init__()

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):

        # x = [sent len, batch size, emb dim]
        #x = torch.tensor(x).float()
        x = torch.cuda.FloatTensor(x)
        output, (hidden, cell) = self.lstm(x)

        # output = [sent len, batch size, hid dim * num directions]
        # hidden = [num layers * num directions, batch size, hid dim]
        # cell = [num layers * num directions, batch size, hid dim]

        # concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
        # and apply dropout

        #hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        # hidden = [batch size, hid dim * num directions]
        hidden = hidden[-1, :, :]
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

        loss.backward()
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


    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = "cuda"
    if device == "cuda":
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    EMBEDDING_DIM = 400
    HIDDEN_DIM = 256
    OUTPUT_DIM = 1

    model = RNN(EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM)
    model.float()

    optimizer = optim.Adam(model.parameters(), lr=0.00001)

    criterion = nn.BCEWithLogitsLoss()
    model = model.to(device)
    criterion = criterion.to(device)

    N_EPOCHS = 5000

    train_raw, val_raw, test = DataLoader.get_data_in_batches()
    train_loader = BatchLoader(train_raw)
    validation_iterator = BatchLoader(val_raw)

    for epoch in range(N_EPOCHS):
        valid_loss, valid_acc = evaluate(model, validation_iterator, criterion)
        train_loss, train_acc = train(model, train_loader, optimizer, criterion)

        print(f'| Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% | Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc*100:.2f}% |')

        # test_loss, test_acc = evaluate(model, test_iterator, criterion)
        #
        # print(f'| Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}% |')