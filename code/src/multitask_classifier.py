import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import math
import gc
from utils.mlt import load_data, DNADataset, train, compute_metrics, one_hot_encode_pad, stratified_split #, count_parameters

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

class MultiTaskTransformerClassifier(nn.Module):
    def __init__(self, n_tokens, n_classes_task1, n_classes_task2, d_model, nhead, num_encoder_layers, dim_feedforward, dropout, max_len):
        super(MultiTaskTransformerClassifier, self).__init__()
        self.embedding = nn.Linear(n_tokens, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        self.decoder_task1 = nn.Linear(d_model, n_classes_task1)
        self.decoder_task2 = nn.Linear(d_model, n_classes_task2)

    def forward(self, src):
        src = self.embedding(src)
        src = src.permute(1, 0, 2)  
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = output.permute(1, 0, 2)  
        
        output_task1 = self.decoder_task1(output[:, -1, :])  
        output_task2 = self.decoder_task2(output[:, -1, :])  
        
        return output_task1, output_task2

# Load your dataset using the provided load_data function
dataset_filename = './src/data/sampred_multiclass_dataset.fa'  
dna_seq, label1, label2 = load_data(dataset_filename)

# Data preprocessing
k = 2
data = one_hot_encode_pad(dna_seq, k)
max_len = data[0].shape[0]
n_tokens = 4**k 

# Define common parameters
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
input_size = 4 ** k
epochs = 1000
patience_cnt = 40
n_classes_task1 = len(np.unique(label1))
n_classes_task2 = len(np.unique(label2))

learning_rate = 0.001

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

test_size = 0.2
val_size = 0.2

def main():
    acc_all_task1 = []
    prec_all_task1 = []
    rec_all_task1 = []
    f1_all_task1 = []

    acc_all_task2 = []
    prec_all_task2 = []
    rec_all_task2 = []
    f1_all_task2 = []
    auc_all_task1 = []
    auc_all_task2 = []

    for i in range(10):
        X_train, X_test, X_val, y_train1, y_train2, y_test1, y_test2, y_val1, y_val2 = stratified_split(data, label1, label2, test_size, val_size, i)

        train_dataset = DNADataset(X_train, y_train1, y_train2)
        val_dataset = DNADataset(X_val, y_val1, y_val2)
        test_dataset = DNADataset(X_test, y_test1, y_test2)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        model = MultiTaskTransformerClassifier(n_tokens, n_classes_task1, n_classes_task2, d_model, nhead, nlayers, dim_feedforward, dropout, max_len).to(device)
        # print(count_parameters(model))

        model = train(model, train_loader, val_loader, epochs, learning_rate, patience_cnt, Lambda, device=device)

        model.load_state_dict(torch.load('best_multitask_model.pth'))

        results = compute_metrics(model, test_loader, device=device)

        acc_all_task1.append(results['task1_accuracy'])
        prec_all_task1.append(results['task1_precision'])
        rec_all_task1.append(results['task1_recall'])
        f1_all_task1.append(results['task1_f1'])
        auc_all_task1.append(results['task1_auc'])

        acc_all_task2.append(results['task2_accuracy'])
        prec_all_task2.append(results['task2_precision'])
        rec_all_task2.append(results['task2_recall'])
        f1_all_task2.append(results['task2_f1'])
        auc_all_task2.append(results['task2_auc'])

        torch.cuda.empty_cache()
        gc.collect()

    print("Task 1 Metrics:")
    print_summary_metrics(acc_all_task1, prec_all_task1, rec_all_task1, f1_all_task1, auc_all_task1)
    print('')
    print("Task 2 Metrics:")
    print_summary_metrics(acc_all_task2, prec_all_task2, rec_all_task2, f1_all_task2, auc_all_task2)

def get_metrics_summary(results, task_name):
    return (
        f"Accuracy: {results[f'{task_name}_accuracy']:.4f} - "
        f"Precision: {results[f'{task_name}_precision']:.4f} - "
        f"Recall: {results[f'{task_name}_recall']:.4f} - "
        f"F1-score: {results[f'{task_name}_f1']:.4f}"
        f"AUC: {results[f'{task_name}_AUC']:.4f}"
    )

def print_summary_metrics(acc_all, prec_all, rec_all, f1_all, auc_all):
    print("Mean Accuracy: {:.4f} ± {:.4f}".format(np.mean(acc_all), np.std(acc_all)))
    print("Mean Precision: {:.4f} ± {:.4f}".format(np.mean(prec_all), np.std(prec_all)))
    print("Mean Recall: {:.4f} ± {:.4f}".format(np.mean(rec_all), np.std(rec_all)))
    print("Mean F1 Score: {:.4f} ± {:.4f}".format(np.mean(f1_all), np.std(f1_all)))
    print("Mean AUC: {:.4f} ± {:.4f}".format(np.mean(auc_all), np.std(auc_all)))

# Define a list of Lambda values you want to iterate over
lambda_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

def run_for_lambda(Lambda):
    print(f"Running for Lambda = {Lambda}")
    main()  # Call the main function
    print('')
    
# Iterate over each Lambda value and run the main function for each Lambda
for Lambda in lambda_values:
    run_for_lambda(Lambda)

