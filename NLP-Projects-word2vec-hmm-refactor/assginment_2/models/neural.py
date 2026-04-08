
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import random


class NER(Dataset):
    def __init__(self,sentences,tags,embeddings,word2idx,window_size=3,unk_vector=None,is_train=True,word_dropout_rate=0.05):
        self.features = []
        self.labels = []

        pad_vector = np.zeros(embeddings.shape[1])
        if unk_vector is None:
            unk_vector = np.zeros(embeddings.shape[1])
        half_window = window_size // 2

        for sentence,tag_seq in zip(sentences,tags):
            embeds = []
            for word in sentence:
                if is_train and random.random() < word_dropout_rate:
                    embeds.append(unk_vector)
                elif word in word2idx:
                    embeds.append(embeddings[word2idx[word]])
                else:
                    embeds.append(unk_vector)
            padded_embeds = [pad_vector]*half_window + embeds + [pad_vector]*half_window
            for i in range(len(sentence)):
                window = padded_embeds[i:i+window_size]
                self.features.append(np.concatenate(window))
                self.labels.append(tag_seq[i])

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self,idx):
        return torch.tensor(self.features[idx],dtype=torch.float32),torch.tensor(self.labels[idx],dtype=torch.long)


def predict_tags(sentence, model, embeddings, word2idx, label_names, window_size=3,unk_vector=None):
    model.eval()
    device = next(model.parameters()).device
    embedding_dim = embeddings.shape[1]
    pad_vector = np.zeros(embedding_dim)
    half_window = window_size // 2
    embeds = [embeddings[word2idx[word]] if word in word2idx else unk_vector for word in sentence]
    padded_embeds = [pad_vector] * half_window + embeds + [pad_vector] * half_window
    features = []
    for i in range(len(sentence)):
        window = padded_embeds[i:i + window_size]
        features.append(np.concatenate(window))
    if not features:
        return []
     
    features_tensor = torch.tensor(np.array(features), dtype=torch.float32).to(device)
    with torch.no_grad():
        outputs = model(features_tensor)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
    return [label_names[p] for p in preds]
        

class FeedForwardNN(nn.Module):
    def __init__(self,embedding_dim,window_size = 3,hidden_dim = 128 ,num_classes = 9,dropout_rate=0.3):
        super(FeedForwardNN,self).__init__()
        input_dim = embedding_dim * window_size
        self.dropout = nn.Dropout(dropout_rate)
        self.layer1 = nn.Linear(input_dim,hidden_dim)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim,num_classes)
    def forward(self,x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x

def evaluate_model(model, dataloader, device, *_, **__):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    print(f"Accuracy: {acc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
    return acc, precision, recall, f1, f1

    
    

    
