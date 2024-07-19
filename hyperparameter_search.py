from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, roc_auc_score
import optuna
import numpy as np
import torch
import pandas as pd

# Load the tokenizer and model
model_name = 'DeepChem/ChemBERTa-5M-MLM'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Tokenization and dataset preparation
def tokenize(smiles_list):
    if not smiles_list:
        print("Received empty smiles list.")
        return {"input_ids": [], "attention_mask": []}
    return tokenizer(smiles_list, padding=True, truncation=True, max_length=128)

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

# Optuna objective function
def objective(trial):
    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy="epoch",
        learning_rate=trial.suggest_loguniform('learning_rate', 1e-5, 5e-5),
        per_device_train_batch_size=trial.suggest_categorical('per_device_train_batch_size', [8, 16, 32]),
        num_train_epochs=trial.suggest_int('num_train_epochs', 3, 5),
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
    )
    
    trainer = Trainer(
        model_init=lambda: AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2),
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )
    
    trainer.train()
    eval_results = trainer.evaluate()
    
    return eval_results["eval_f1"]

study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
study.optimize(objective, n_trials=10)

print("Best trial:")
trial = study.best_trial
print(f"  Value: {trial.value}")
for key, value in trial.params.items():
    print(f"    {key}: {value}")