from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, roc_auc_score
import numpy as np
import torch
import pandas as pd

# Load the tokenizer and model
model_name = "DeepChem/ChemBERTa-5M-MLM"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Tokenization and dataset preparation
def tokenize(smiles_list):
    if not smiles_list:
        print("Received empty smiles list.")
        return {"input_ids": [], "attention_mask": []}
    return tokenizer(smiles_list, padding=True, truncation=True, max_length=512)

train_df = pd.read_csv('./Downstream_tasks/bbbp_scaffold_train.csv')
val_df = pd.read_csv('./Downstream_tasks/bbbp_scaffold_valid.csv')
test_df = pd.read_csv('./Downstream_tasks/bbbp_scaffold_test.csv')

train_encodings = tokenize(train_df['smiles'].tolist())
val_encodings = tokenize(val_df['smiles'].tolist())
test_encodings = tokenize(test_df['smiles'].tolist())

class BBBPDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = BBBPDataset(train_encodings, list(train_df['p_np']))
val_dataset = BBBPDataset(val_encodings, list(val_df['p_np']))
test_dataset = BBBPDataset(test_encodings, list(test_df['p_np']))

# Define compute metrics function
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(p.label_ids, preds, average='binary')
    cm = confusion_matrix(p.label_ids, preds)
    roc_auc = roc_auc_score(p.label_ids, p.predictions[:, 1])
    print("Confusion Matrix:")
    for row in cm:
        print(row)
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc
    }

# Training arguments with specified hyperparameters
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=4.3072444502824424e-05,
    per_device_train_batch_size=16,
    num_train_epochs=5,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss"
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Evaluate the model on the test dataset
eval_results = trainer.evaluate(eval_dataset=test_dataset)

print(f"Test Precision: {eval_results['eval_precision']}")
print(f"Test Recall: {eval_results['eval_recall']}")
print(f"Test F1-Score: {eval_results['eval_f1']}")
print(f"Test ROC-AUC: {eval_results['eval_roc_auc']}")
