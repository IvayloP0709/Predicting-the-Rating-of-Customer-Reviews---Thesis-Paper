import torch
import torch.nn as nn
from transformers import AdamW, RobertaForSequenceClassification, RobertaModel
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

class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, num_extra_dims):
        super().__init__()
        total_dims = config.hidden_size+num_extra_dims
        self.dense = nn.Linear(total_dims, total_dims)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(total_dims, config.num_labels)

    def forward(self, features, **kwargs):
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class CustomSequenceClassification(RobertaForSequenceClassification):

    def __init__(self, config, num_extra_dims):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        # might need to rename this depending on the model
        self.roberta =  RobertaModel(config)
        self.classifier = ClassificationHead(config, num_extra_dims)

        # Initialize weights and apply final processing
        self.post_init()


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        extra_data: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # sequence_output will be (batch_size, seq_length, hidden_size)
        sequence_output = outputs[0]

        # additional data should be (batch_size, num_extra_dims)
        cls_embedding = sequence_output[:, 0, :]

        output = torch.cat((cls_embedding, extra_data), dim=-1)

        logits = self.classifier(output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

# Read data
df = pd.read_csv('neg_preprocessed_clean_y.csv')

# Train, validation, and test split
X = df[['text', 'resp_text']]
y = df['rating']
ordinal_mapping = {1.0: 0, 2.0: 1, 3.0: 2, 4.0: 3, 5.0: 4}
y = y.map(ordinal_mapping)
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, stratify=y_train_val, random_state=42)  

# Use oversampling to balance the training set
oversampler = RandomOverSampler(random_state=42)
X_train, y_train = oversampler.fit_resample(X_train, y_train)

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
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

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

# Additional features
selected_columns = ['response_timing', 'response_length', 'general_personal', 'price_range']
selected_features_df = df[selected_columns]
additional_features_array = selected_features_df.values
additional_features_tensor = torch.tensor(additional_features_array, dtype=torch.float)

# Model
model = CustomSequenceClassification.from_pretrained('roberta-base', num_extra_dims=4, num_labels=5, problem_type='multi_label_classification')

# DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Hyperparameters and optimization
learning_rate = 4e-5
epochs = 4
dropout = 0.1
accumulation_steps = 1
weight_decay = 0.01

optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)  # Add weight decay to optimizer

model.to(device)
criterion = nn.CrossEntropyLoss()
patience = 5
min_val_loss = float('inf')
early_stopping_counter = 0

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for i, (input_ids, attention_mask, labels) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}')):
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
        additional_features_batch = additional_features_tensor[:len(input_ids)].to(device)

        output = model(input_ids, attention_mask=attention_mask, extra_data=additional_features_batch)
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
        additional_features_batch = additional_features_tensor[:len(input_ids)].to(device)

        with torch.no_grad():
            output = model(input_ids, attention_mask=attention_mask, extra_data=additional_features_batch)
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

# DataLoader for Test
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Lists to store predictions and labels
all_preds = []
all_labels = []

# Iterate through test set batches
for input_ids, attention_mask, labels in tqdm(test_loader, desc='Testing'):
    input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
    additional_features_batch = additional_features_tensor[:len(input_ids)].to(device)

    # Forward pass
    with torch.no_grad():
        output = model(input_ids, attention_mask=attention_mask, extra_data=additional_features_batch)
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
plt.show()
plt.savefig('confusion_matrix_rob_add.png', dpi=300)

print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1}")