from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
import torch
import numpy as np
import pandas as pd
from transformers import get_scheduler
from transformers import AdamW

# Load the tokenizer and model
model_name = 'DeepChem/ChemBERTa-5M-MLM'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Configure and add adapter
lora_config = LoraConfig(
    r=62,  # Best trial r
    lora_alpha=90,  # Best trial lora_alpha
    lora_dropout=0.2100331665475732,  # Best trial lora_dropout
    bias="none",
    task_type="SEQ_CLS"  # for sequence classification
)

model = get_peft_model(model, lora_config)

# Split the data
train_df = pd.read_csv('./Downstream_tasks/bbbp_scaffold_train.csv')
val_df = pd.read_csv('./Downstream_tasks/bbbp_scaffold_valid.csv')
test_df = pd.read_csv('./Downstream_tasks/bbbp_scaffold_test.csv')


def tokenize(smiles_list):
    if not smiles_list:
        print("Received empty smiles list.")
        return {"input_ids": [], "attention_mask": []}

    return tokenizer(smiles_list, padding=True, truncation=True, max_length=128)


train_encodings = tokenize(train_df['smiles'].tolist())
val_encodings = tokenize(val_df['smiles'].tolist())
test_encodings = tokenize(test_df['smiles'].tolist())

# Create PyTorch datasets
class BBBPDataset(torch.utils.data.Dataset):
  def __init__(self, encodings, labels):
    self.encodings = encodings
    #self.labels = labels
    self.labels = torch.tensor(labels, dtype=torch.long)

  def __getitem__(self, idx):
    item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
    #item['labels'] = torch.tensor(self.labels[idx])
    item['labels'] = self.labels[idx] 
    return item

  def __len__(self):
    return len(self.labels)

train_dataset = BBBPDataset(train_encodings, list(train_df['p_np']))
val_dataset = BBBPDataset(val_encodings, list(val_df['p_np']))
test_dataset = BBBPDataset(test_encodings, list(test_df['p_np']))

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=25,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
)

# Define the compute metrics function
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(p.label_ids, preds, average='binary')
    cm = confusion_matrix(p.label_ids, preds)
    roc_auc = roc_auc_score(p.label_ids, p.predictions[:, 1])

    # Ensure this print statement works as expected
    print("Confusion Matrix:")
    print(cm)

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc
    }

def print_confusion_matrix(cm):
    print("Confusion Matrix:")
    for row in cm:
        print(row)


# Example to create a scheduler manually (usually integrated within Trainer via TrainingArguments)
optimizer = AdamW(model.parameters(), lr=5e-5)
scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=500,
    num_training_steps=len(train_dataset) // training_args.per_device_train_batch_size * training_args.num_train_epochs
)

# Add scheduler to your trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Evaluate the model
eval_results = trainer.evaluate(eval_dataset=test_dataset)

print(f"Test Precision: {eval_results['eval_precision']}")
print(f"Test Recall: {eval_results['eval_recall']}")
print(f"Test F1-Score: {eval_results['eval_f1']}")
print(f"Test ROC-AUC: {eval_results['eval_roc_auc']}")