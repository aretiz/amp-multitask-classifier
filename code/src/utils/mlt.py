import itertools
import torch
from torch.utils.data import Dataset
import time
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


class DNADataset(Dataset):
    def __init__(self, data, label1, label2):
        self.data = data
        self.label1 = label1
        self.label2 = label2

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.label1[idx], self.label2[idx]


def load_data(dataset):
    data = []
    label1 = []
    label2 = []
    with open(dataset, 'r') as f:
        for line in f:
            seq, l1, l2 = line.split()
            data += [seq]
            label1 += [int(l1)]
            label2 += [int(l2)]
    return data, label1, label2


def stratified_split(X, y, y2, test_size, val_size, random_state):
    X_train, X_test, y_train, y_test1, train_index, test_index = train_test_split(X, y, range(len(y)), test_size=test_size,
                                                                                  stratify=y, random_state=random_state)
    X_train, X_val, y_train1, y_val1, train_index, val_index = train_test_split(X_train, y_train, train_index,
                                                                                test_size=val_size, stratify=y_train,
                                                                                random_state=random_state)
    y_train2 = [y2[i] for i in train_index]
    y_val2 = [y2[i] for i in val_index]
    y_test2 = [y2[i] for i in test_index]

    return X_train, X_test, X_val, y_train1, y_train2, y_test1, y_test2, y_val1, y_val2

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def one_hot_encode_pad(sequences, n):
    n_words = [''.join(n_word) for n_word in itertools.product('ACGT', repeat=n)]

    # Create a dictionary of n-words and their corresponding one-hot-encoding
    n_word_dict = {n_word: i for i, n_word in enumerate(n_words)}

    # Find the maximum length of all sequences
    max_len = max([len(seq) for seq in sequences])

    all_one_hot = []
    for seq in sequences:
        seq_len = len(seq)
        one_hot = torch.zeros(max_len, len(n_word_dict))

        # Fill in the one-hot-encoding for the given sequence
        for i in range(seq_len - n + 1):
            n_word = seq[i:i + n]
            one_hot[i, n_word_dict[n_word]] = 1

        all_one_hot.append(one_hot)

    return all_one_hot


def compute_test(model, loader, weight, device):
    loss_fn1 = nn.CrossEntropyLoss()
    loss_fn2 = nn.CrossEntropyLoss()

    total_correct = 0.0
    total_loss = 0.0
    total = 0.0
    correct1 = 0.0
    correct2 = 0.0
    loss1 = 0.0
    loss2 = 0.0

    for i, (x, target1, target2) in enumerate(loader):
        x = x.to(device)
        target1 = target1.to(device)
        target2 = target2.to(device)

        output1, output2 = model(x)
        pred1 = output1.max(dim=1)[1]  # get the index of the max log-probability
        correct1 += pred1.eq(target1).sum().item()
        loss1 += loss_fn1(output1, target1).mean()

        pred2 = output2.max(dim=1)[1]  # get the index of the max log-probability
        correct2 += pred2.eq(target2).sum().item()
        loss2 += loss_fn2(output2, target2).mean()

        combined_outputs = torch.cat([output1, output2], dim=1)
        pred = combined_outputs.max(dim=1)[1]
        total_correct += (pred == target1 + 2 * target2).sum().item()
        total += target1.size(0)

        l = loss_fn1(output1, target1) + weight * loss_fn2(output2, target2)
        total_loss += l.item()

    return correct1 / len(loader.dataset), loss1, correct2 / len(loader.dataset), loss2, total_correct / total, \
           total_loss / len(loader)


def train(model, train_loader, val_loader, epochs, lr, patience, weight, device):
    min_loss = 1e10
    patience_cnt = 0
    val_loss_values = []
    best_epoch = 0

    loss_fn1 = nn.CrossEntropyLoss()
    loss_fn2 = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # print("Starting training...")
    t = time.time()
    model.train()
    model.to(device)

    for epoch in range(epochs):
        loss_train = 0.0
        correct1 = 0
        correct2 = 0
        for i, (x, target1, target2) in enumerate(train_loader):
            x = x.to(device)
            target1 = target1.to(device)
            target2 = target2.to(device)
            output1, output2 = model(x)

            loss1 = loss_fn1(output1, target1)
            loss2 = loss_fn2(output2, target2)
            loss = loss1 + weight*loss2

            loss.backward()
            optimizer.step()  # do gradient descent over the batch
            optimizer.zero_grad()  # clear the gradient

            loss_train += loss.item()
            pred1 = output1.max(dim=1)[1]  # get the index of the max log-probability
            pred2 = output2.max(dim=1)[1]
            correct1 += pred1.eq(target1).sum().item()
            correct2 += pred2.eq(target2).sum().item()

        acc_train1 = correct1 / len(train_loader.dataset)
        acc_train2 = correct2 / len(train_loader.dataset)

        with torch.no_grad():
            acc_val1, loss_val1, acc_val2, loss_val2, acc_val, loss_val = compute_test(model, val_loader, weight, device)

        val_loss_values.append(loss_val)

        if val_loss_values[-1] < min_loss:
            min_loss = val_loss_values[-1]
            best_epoch = epoch
            patience_cnt = 0
            torch.save(model.state_dict(), 'best_multitask_model.pth')
        else:
            patience_cnt += 1

        if patience_cnt == patience:
            # print(epoch + 1)
            break

    # print('Optimization Finished! Total time elapsed: {:.6f}'.format(time.time() - t))

    return model

def compute_metrics(model, data_loader, device):
    model.eval()
    model.to(device)

    y_true1, y_pred1, y_true2, y_pred2 = [], [], [], []

    with torch.no_grad():
        for x, target1, target2 in data_loader:
            x, target1, target2 = x.to(device), target1.to(device), target2.to(device)

            out1, out2 = model(x)
            y_pred1.append(torch.sigmoid(out1)[:, 1].cpu().numpy())  # Assuming 1 is the positive class
            y_true1.append(target1.cpu().numpy())
            y_pred2.append(torch.sigmoid(out2)[:, 1].cpu().numpy())
            y_true2.append(target2.cpu().numpy())

    y_pred1 = np.concatenate(y_pred1)
    y_true1 = np.concatenate(y_true1)
    y_pred2 = np.concatenate(y_pred2)
    y_true2 = np.concatenate(y_true2)

    threshold = 0.5
    y_pred1_binary = (y_pred1 > threshold).astype(int)
    y_pred2_binary = (y_pred2 > threshold).astype(int)

    accuracy1 = accuracy_score(y_true1, y_pred1_binary)
    precision1 = precision_score(y_true1, y_pred1_binary)
    recall1 = recall_score(y_true1, y_pred1_binary)
    f1_1 = f1_score(y_true1, y_pred1_binary)
    auc1 = roc_auc_score(y_true1, y_pred1)

    accuracy2 = accuracy_score(y_true2, y_pred2_binary)
    precision2 = precision_score(y_true2, y_pred2_binary)
    recall2 = recall_score(y_true2, y_pred2_binary)
    f1_2 = f1_score(y_true2, y_pred2_binary)
    auc2 = roc_auc_score(y_true2, y_pred2)

    results = {
        'task1_accuracy': accuracy1,
        'task1_precision': precision1,
        'task1_recall': recall1,
        'task1_f1': f1_1,
        'task1_auc': auc1,
        'task2_accuracy': accuracy2,
        'task2_precision': precision2,
        'task2_recall': recall2,
        'task2_f1': f1_2,
        'task2_auc': auc2
    }

    return results

