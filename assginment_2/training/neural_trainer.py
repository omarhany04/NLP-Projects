import itertools

import torch

from models.neural import NER, FeedForwardNN, evaluate_model

import copy
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
def train_neural_model(train_sentences,train_tags,val_sentences,val_tags,embeddings,word2idx,label_names,epochs = 20):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("Tuning hyperparameters...")
    grid = {
        "hidden_dim": [32, 64, 128],
        "learning_rate": [0.001, 0.0005],
        "window_size": [3, 5],
        "dropout_rate": [0.1, 0.3, 0.5],
        "weight_decay": [1e-4, 1e-5],
    }

    key = grid.keys()
    combinations = list(itertools.product(*grid.values()))

    best_f1 = 0.0
    best_params = None
    best_model_state = None

    unk_vector = np.mean(embeddings, axis=0)

    num_classes = len(label_names)
    embedding_dim = embeddings.shape[1]
    batch_size = 64

    for combo in combinations:
        params = dict(zip(key, combo))
        print(f"Testing combination: {params}")

        train_dataset = NER(train_sentences, train_tags, embeddings=embeddings, word2idx=word2idx, window_size=params["window_size"],unk_vector=unk_vector,is_train=True, word_dropout_rate=0.05)
        val_dataset = NER(val_sentences, val_tags, embeddings=embeddings, word2idx=word2idx, window_size=params["window_size"],unk_vector=unk_vector,is_train=False)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

        model = FeedForwardNN(embedding_dim=embedding_dim,window_size=params["window_size"],hidden_dim=params["hidden_dim"],num_classes=num_classes,dropout_rate=params["dropout_rate"])
        model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"])

        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for features,labels in train_loader:
                features = features.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
        print("Evaluating on Validation Set...")
        _, _, _, _, val_f1 = evaluate_model(model, val_loader, device,criterion, label_names)
        
        if val_f1 > best_f1:
            print(f"*** New Best F1-Score: {val_f1:.4f}! ***")
            best_f1 = val_f1
            best_params = params
            # Save the weights of the best model
            best_model_state = copy.deepcopy(model.state_dict())

    print("TUNING COMPLETE")
    print(f"Best Parameters: {best_params}")
    print(f"Best Validation F1: {best_f1:.4f}")
    
    return best_params, best_model_state

    

