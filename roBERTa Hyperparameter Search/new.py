import pandas as pd 

df = pd.read_csv('neg_preprocessed_clean_y.csv')

df.info()

# Train, validation, and test split
from sklearn.model_selection import train_test_split

X = df[['text', 'resp_text']]
y = df['rating']

# Ordinal encoding of the 'rating' variable
ordinal_mapping = {1.0: 0, 2.0: 1, 3.0: 2, 4.0: 3, 5.0: 4}
y = y.map(ordinal_mapping)

X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, stratify=y_train_val, random_state=42)  # 0.25 x 0.8 = 0.2

# Convert 'text' and 'resp_text' columns to strings
X_train['text_str'] = [str(i) for i in X_train['text'].values]
X_train['resp_text_str'] = [str(i) for i in X_train['resp_text'].values]

X_val['text_str'] = [str(i) for i in X_val['text'].values]
X_val['resp_text_str'] = [str(i) for i in X_val['resp_text'].values]

X_test['text_str'] = [str(i) for i in X_test['text'].values]
X_test['resp_text_str'] = [str(i) for i in X_test['resp_text'].values]

# Combine 'text' and 'resp_text' into a single column
X_train_combined = X_train['text_str'] + ' ' + X_train['resp_text_str']
X_val_combined = X_val['text_str'] + ' ' + X_val['resp_text_str']
X_test_combined = X_test['text_str'] + ' ' + X_test['resp_text_str']

# Tokenizer
from transformers import RobertaTokenizer
import torch

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# Encode the combined text
def encode_text(text):
    return tokenizer(text, padding=True, truncation=True, return_tensors='pt')

# Encode the combined text for train, validation, and test sets
encoded_train = encode_text(X_train_combined.tolist())
encoded_val = encode_text(X_val_combined.tolist())
encoded_test = encode_text(X_test_combined.tolist())

# Create datasets
from torch.utils.data import TensorDataset

train_dataset = TensorDataset(encoded_train['input_ids'], encoded_train['attention_mask'], torch.tensor(y_train.tolist()))
val_dataset = TensorDataset(encoded_val['input_ids'], encoded_val['attention_mask'], torch.tensor(y_val.tolist()))
test_dataset = TensorDataset(encoded_test['input_ids'], encoded_test['attention_mask'], torch.tensor(y_test.tolist()))

import optuna
import torch
import torch.nn as nn
from transformers import RobertaForSequenceClassification, AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

def objective(trial):
    learning_rate = trial.suggest_categorical('learning_rate',[2e-5, 3e-5, 5e-5])
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    epochs = trial.suggest_int('epochs', low=2, high=4)
    dropout = trial.suggest_float('dropout', low=0.0, high=0.5, step=0.1)
    accumulation_steps = trial.suggest_int('accumulation_steps', low=1, high=4)

    model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=5,
                                                           problem_type='multi_label_classification')
    classifier = nn.Linear(model.config.hidden_size, 5)
    classifier_dropout = nn.Dropout(dropout)
    classifier = nn.Sequential(classifier_dropout, classifier)
    model.classifier = classifier

    optimizer = AdamW(model.parameters(), lr=learning_rate)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()
        for i, (input_ids, attention_mask, labels) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}')):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            output = model(input_ids, attention_mask=attention_mask)
            loss = criterion(output.logits, labels)
            loss = loss / accumulation_steps  # Scale loss for gradient accumulation
            loss.backward()
            total_loss += loss.item()
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        model.eval()
        val_losses = []
        val_preds = []
        val_labels = []
        for input_ids, attention_mask, labels in tqdm(val_loader, desc=f'Epoch {epoch + 1}/{epochs} Validation'):
            with torch.no_grad():
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)
                output = model(input_ids, attention_mask=attention_mask)
                logits = output.logits.float()
                loss = criterion(logits, labels)
                val_losses.append(loss.item())
                val_preds.extend(logits.argmax(dim=1).tolist())
                val_labels.extend(labels.tolist())

        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average='weighted')

        print(f'Train Loss: {total_loss / len(train_loader)}')
        print(f'Val Loss: {sum(val_losses) / len(val_losses)}')
        print(f'Val Acc: {val_acc}')
        print(f'Val F1: {val_f1}')
    return val_f1

study = optuna.create_study(direction='maximize', sampler=optuna.samplers.RandomSampler())
study.optimize(objective, n_trials=10)
best_params = study.best_params
print(best_params)