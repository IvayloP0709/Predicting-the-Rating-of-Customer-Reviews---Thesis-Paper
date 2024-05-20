import torch
import torch.nn as nn
from transformers import BertModel, AdamW, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from imblearn.over_sampling import RandomOverSampler
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers.modeling_outputs import SequenceClassifierOutput
from typing import Optional, Tuple, Union
import seaborn as sns
import matplotlib.pyplot as plt




# Read data
df = pd.read_csv('neg_preprocessed_clean_y.csv')

# Train, validation, and test split
X = df[['text', 'resp_text']]
y = df['rating']
ordinal_mapping = {1.0: 0, 2.0: 1, 3.0: 2, 4.0: 3, 5.0: 4}
y = y.map(ordinal_mapping)
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, stratify=y_train_val, random_state=42)  

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
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Encode the combined text
def encode_text(text):
    return tokenizer(text, padding=True, truncation=True, return_tensors='pt')

# Encode the combined text for train, validation, and test sets
encoded_train = encode_text(X_train_combined.tolist())
encoded_val = encode_text(X_val_combined.tolist())
encoded_test = encode_text(X_test_combined.tolist())

# Create datasets
train_dataset = TensorDataset(encoded_train['input_ids'], encoded_train['attention_mask'], torch.tensor(y_train.tolist()))
val_dataset = TensorDataset(encoded_val['input_ids'], encoded_val['attention_mask'], torch.tensor(y_val.tolist()))
test_dataset = TensorDataset(encoded_test['input_ids'], encoded_test['attention_mask'], torch.tensor(y_test.tolist()))

# FOR THE DATA BALANCING
import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight

def compute_class_weights(y_train):
    """
    Compute class weights given imbalanced training data.
    Usually used in the neural network model to augment the loss function (weighted loss function),
    favoring/giving more weights to the rare classes.
    """
    class_list = np.unique(y_train)
    class_weight_value = compute_class_weight('balanced', classes=class_list, y=y_train)
    class_weight = dict()

    # Build the dictionary using the weight obtained from the scikit function
    for i, class_value in enumerate(class_list):
        class_weight[class_value] = class_weight_value[i]

    return torch.FloatTensor([class_weight[i] for i in y_train])

# Example usage:
class_weights = compute_class_weights(y_train)

# Additional features
selected_columns = ['response_timing', 'response_length', 'general_personal', 'price_range']
selected_features_df = df[selected_columns]
additional_features_array = selected_features_df.values
additional_features_tensor = torch.tensor(additional_features_array, dtype=torch.float)

# Model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=5, problem_type='multi_label_classification')

# DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Hyperparameters and optimization
learning_rate = 5e-5
epochs = 4
dropout = 0.5
accumulation_steps = 3
weight_decay = 0.01

optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)  # Add weight decay to optimizer

model.to(device)

# Convert class weights to tensor
num_classes = 5
weight_tensor = torch.FloatTensor([class_weights[i] for i in range(num_classes)])

# Ensure that the weight tensor has the correct shape for the number of classes
if len(weight_tensor) != num_classes:
    raise ValueError(f"The weight tensor shape {len(weight_tensor)} does not match the number of classes {num_classes}")

# Define the loss function with weighted loss
criterion = nn.CrossEntropyLoss(weight=weight_tensor).to(device)

patience = 2
min_val_loss = float('inf')
early_stopping_counter = 0

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for i, (input_ids, attention_mask, labels) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}')):
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

        output = model(input_ids, attention_mask=attention_mask)
        loss = criterion(output.logits, labels)
        loss = loss / accumulation_steps  
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
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

        with torch.no_grad():
            output = model(input_ids, attention_mask=attention_mask)
            logits = output.logits.float()
            loss = criterion(logits, labels)
            val_losses.append(loss.item())
            val_preds.extend(logits.argmax(dim=1).tolist())
            val_labels.extend(labels.tolist())

    val_loss = sum(val_losses) / len(val_losses)
    val_acc = accuracy_score(val_labels, val_preds)
    val_f1 = f1_score(val_labels, val_preds, average='weighted')
    print(f'Train Loss: {total_loss / len(train_loader)}')
    print(f'Val Loss: {val_loss}')
    print(f'Val Acc: {val_acc}')
    print(f'Val F1: {val_f1}')

    if val_loss < min_val_loss:
        min_val_loss = val_loss
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= patience:
            print("Early stopping triggered!")
            break

# Lists to store predictions and labels
all_preds = []
all_labels = []

# Iterate through test set batches
for input_ids, attention_mask, labels in tqdm(test_loader, desc='Testing'):
    input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

    # Forward pass
    with torch.no_grad():
        output = model(input_ids, attention_mask=attention_mask)
        logits = output.logits
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

# Calculate accuracy and F1 score
accuracy = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds, average='weighted')

# Generate confusion matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=ordinal_mapping.values(), yticklabels=ordinal_mapping.values())
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix_bert_weight.png', dpi=300)
plt.show()

print(f"Test Accuracy: {accuracy}")
print(f"Test F1 Score: {f1}")