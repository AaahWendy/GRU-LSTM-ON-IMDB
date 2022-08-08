from datetime import time

import torch
import torchtext
from torch import nn, optim
from torchtext import data
from torchtext import datasets
import random
import math
import numpy as np
import torch
# from LSTMmodel import LSTM
import matplotlib.pyplot  as plt
from torch import nn as nn
from torch import Tensor
from typing import Tuple

import torch.nn as nn
MAX_WORD = 10000  # 只保留最高频的10000词
MAX_LEN = 300     # 句子统一长度为200
word_count={}     # 词-词出现的词数 词典

TEXT = data.Field(tokenize='spacy',tokenizer_language='en_core_web_sm',include_lengths = True)
LABEL = data.LabelField(dtype=torch.float)

train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
train_data, valid_data = train_data.split(0.8)
print (torch.cuda.is_available())
print(f'Number of training examples: {len(train_data)}')
print(f'Number of val examples: {len(valid_data)}')
print(f'Number of testing examples: {len(test_data)}')

MAX_VOCAB_SIZE = 25_000
TEXT.build_vocab(train_data,vectors='glove.6B.100d',max_size=MAX_VOCAB_SIZE)
LABEL.build_vocab(train_data)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
BATCH_SIZE = 32
N_EPOCHS = 10
train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data,valid_data,test_data),
    batch_size=BATCH_SIZE,device=device)


# class RNN(nn.Module):
#     def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
#         super().__init__()
#         self.embedding = nn.Embedding(input_dim, embedding_dim)
#         self.rnn = nn.RNN(embedding_dim, hidden_dim)
#         self.fc = nn.Linear(hidden_dim, output_dim)
#
#     def forward(self, text):
#         # text = [sent len, batch size]
#         embedded = self.embedding(text)
#         # embedded = [sent len, batch size, emb dim]
#         output, hidden = self.rnn(embedded)
#         # output = [sent len, batch size, hid dim]
#         # hidden = [1, batch size, hid dim]
#         assert torch.equal(output[-1, :, :], hidden.squeeze(0))
#         return self.fc(hidden.squeeze(0))
#
# INPUT_DIM = len(TEXT.vocab)
# EMBEDDING_DIM = 100
# HIDDEN_DIM = 256
# OUTPUT_DIM = 1
#
# model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM)
# import torch.optim as optim
# optimizer=optim.SGD(model.parameters(),lr=1e-3,momentum=0.9)
# criterion=nn.BCEWithLogitsLoss()#计算二元交叉熵
# if torch.cuda.is_available():
#     model = model.cuda()
# # model=model.to(device)
# nivicriterion=criterion.to(device)


class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers,
                 bidirectional, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.rnn = nn.LSTM(embedding_dim,
                           hidden_dim,
                           num_layers=n_layers,
                           bidirectional=bidirectional,
                           dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, text, text_lengths):
        # text = [sent len, batch size]
        embedded = self.dropout(self.embedding(text))
        # embedded = [sent len, batch size, emb dim]
        # pack sequence
        # lengths need to be on CPU!
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.to('cpu'))
        packed_output, (hidden, cell) = self.rnn(packed_embedded)
        # unpack sequence
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)
        # output = [sent len, batch size, hid dim * num directions]
        # output over padding tokens are zero tensors
        # hidden = [num layers * num directions, batch size, hid dim]
        # cell = [num layers * num directions, batch size, hid dim]
        # concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
        # and apply dropout
        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        # hidden = [batch size, hid dim * num directions]
        return self.fc(hidden)

INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 1
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.5
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

model = RNN(INPUT_DIM,
            EMBEDDING_DIM,
            HIDDEN_DIM,
            OUTPUT_DIM,
            N_LAYERS,
            BIDIRECTIONAL,
            DROPOUT,
            PAD_IDX)

UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
#
pretrained_embeddings = TEXT.vocab.vectors
model.embedding.weight.data.copy_(pretrained_embeddings)

model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)
#


import torch.optim as optim
optimizer=optim.SGD(model.parameters(),lr=0.01,momentum=0.9)
criterion=nn.BCEWithLogitsLoss()#计算二元交叉熵
if torch.cuda.is_available():
    model = model.cuda()
criterion=criterion.to(device)


def binary_accuracy(preds, y):
    preds = torch.round(torch.sigmoid(preds))
    correct = (preds == y).float()
    return correct.sum() / len(correct)


def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.train()
    for batch in iterator:
        optimizer.zero_grad()
        text, text_lengths = batch.text
        predictions = model(text, text_lengths).squeeze(1)
        # predictions = model(batch.text).squeeze(1)
        loss = criterion(predictions, batch.label)
        acc = binary_accuracy(predictions, batch.label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for batch in iterator:
            text, text_lengths = batch.text
            predictions = model(text, text_lengths).squeeze(1)
            # predictions = model(batch.text).squeeze(1)
            loss = criterion(predictions, batch.label)
            acc = binary_accuracy(predictions, batch.label)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


if __name__ == '__main__':
    trloss=[]
    tracc=[]
    valacc=[]
    valloss=[]
    best_valid_loss = float('inf')

    for epoch in range(N_EPOCHS):
        train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
        print(f'Epoch: {epoch + 1:02} | ')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc* 100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')
        trloss.append(train_loss)
        tracc.append(train_acc)
        valloss.append(valid_loss)
        valacc.append(valid_acc)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'GRU.pth')

    print(trloss)
    print(valloss)
    plt.plot(trloss, color='g', label="trainloss")
    plt.plot(valloss, color='b', label="valloss")
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title(' GRUNet')
    plt.legend()
    plt.show()

    print(tracc)
    print(valacc)
    plt.plot(tracc, color='y', label="trainacc")
    plt.plot(valacc, color='r', label="valacc")
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.title(' GRUNet')
    plt.legend()
    plt.show()

