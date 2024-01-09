import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import math
import gc
from utils.single import load_data, DNADataset, train, compute_metrics, one_hot_encode_pad, stratified_split 

# Set random seeds for reproducibility
seed = 2
torch.manual_seed(seed)
np.random.seed(seed)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerClassifier(nn.Module):
    def __init__(self, n_tokens, n_classes_task1, d_model, nhead, num_encoder_layers, dim_feedforward, dropout, max_len):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Linear(n_tokens, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        self.decoder_task1 = nn.Linear(d_model, n_classes_task1)

    def forward(self, src):
        src = self.embedding(src)
        src = src.permute(1, 0, 2)  
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = output.permute(1, 0, 2)  
        
        output_task1 = self.decoder_task1(output[:, -1, :])  
                 
        return output_task1 


# Load your dataset using the provided load_data function
dataset_filename = './src/data/sampred_multiclass_dataset.fa'
dna_seq, label1, label2  = load_data(dataset_filename)

# Data preprocessing
k = 2
data = one_hot_encode_pad(dna_seq, k)
max_len = data[0].shape[0]

# Define common parameters
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
input_size = 4 ** k
epochs = 1000
patience_cnt = 40
n_classes_task = len(np.unique(label1))
learning_rate = 0.001
test_size = 0.2
val_size = 0.2

# Dataset-specific parameters
if 'AMP_multiclass_dataset.fa' in dataset_filename: # Dataset 1
    batch_size = 64
    d_model = 32
    nhead = 2
    nlayers = 4
    dim_feedforward = 64
    dropout = 0.1
elif 'sampred_multiclass_dataset.fa' in dataset_filename: # Dataset 2
    batch_size = 8
    d_model = 64
    nhead = 4
    nlayers = 2
    dim_feedforward = 32
    dropout = 0.2
else:
    raise ValueError("Unknown dataset filename. Please specify parameters for the given dataset.")

def main():
    # Initialize empty lists to store the results
    acc_all = []
    prec_all = []
    rec_all = []
    f1_all = []
    auc_all = []

    for i in range(10):
        # print(f"Run {i+1}")
        ################ Select the task for classification ###################
                                # label1 --> AMP
                                # label2 --> fold
        X_train, X_test, X_val, y_train, y_test, y_val = stratified_split(data, label1, test_size, val_size, i)

        # dataloaders
        train_dataset = DNADataset(X_train, y_train)
        val_dataset = DNADataset(X_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        model = TransformerClassifier(input_size, n_classes_task, d_model, nhead, nlayers, dim_feedforward, dropout, max_len).to(device)

        model = train(model, train_loader, val_loader, epochs, learning_rate, patience_cnt, device=device)

        model.load_state_dict(torch.load('best_model_single_task.pth'))

        # predict on the test set
        dataset = DNADataset(X_test, y_test)
        test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        accuracy, precision, recall, f1, auc = compute_metrics(model, test_loader, device)

        acc_all.append(accuracy)
        prec_all.append(precision)
        rec_all.append(recall)
        f1_all.append(f1)
        auc_all.append(auc)

        torch.cuda.empty_cache()
        gc.collect()

    # Print the results
    print("Mean Accuracy: {:.4f} ± {:.4f}".format(np.mean(acc_all), np.std(acc_all)))
    print("Mean Precision: {:.4f} ± {:.4f}".format(np.mean(prec_all), np.std(prec_all)))
    print("Mean Recall: {:.4f} ± {:.4f}".format(np.mean(rec_all), np.std(rec_all)))
    print("Mean F1 Score: {:.4f} ± {:.4f}".format(np.mean(f1_all), np.std(f1_all)))
    print("Mean AUC: {:.4f} ± {:.4f}".format(np.mean(auc_all), np.std(auc_all)))

if __name__ == "__main__":
    main() 