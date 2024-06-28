import os
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForMaskedLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling, EvalPrediction, TrainerCallback
import wandb
import optuna
import plotly.graph_objects as vis

# Expandable memory segments configuration for PyTorch
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Set max_split_size_mb to avoid memory fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

# Initialize W&B
wandb.init(project="chemberta-finetuning")

# Set device to GPU if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-5M-MLM")
model = AutoModelForMaskedLM.from_pretrained("DeepChem/ChemBERTa-5M-MLM")
model.to(device)  # Move the model to the specified device

# Load and prepare data
with open('./Datasets/combined_nps.txt', 'r') as file:
    data = file.readlines()
    data = [line.strip() for line in data]
# Convert list to DataFrame to use sample method
data_df = pd.DataFrame(data, columns=['smiles'])
data_df = data_df.sample(frac=0.1, random_state=42)  # Sampling a fraction for demonstration

# Convert DataFrame back to list after sampling
data = data_df['smiles'].tolist()

def encode_smiles(smiles_list):
    return tokenizer(smiles_list, add_special_tokens=True, truncation=True, max_length=512, padding="max_length", return_tensors="pt")

class SMILESDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        return {key: self.encodings[key][idx] for key in self.encodings}

encoded_data = encode_smiles(data)
dataset = SMILESDataset(encoded_data)

# Split data into train, eval, and test sets (80/10/10)
train_idx, temp_idx = train_test_split(range(len(encoded_data['input_ids'])), test_size=0.2, random_state=42)
val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)

train_dataset = SMILESDataset({key: val[train_idx] for key, val in encoded_data.items()})
val_dataset = SMILESDataset({key: val[val_idx] for key, val in encoded_data.items()})
test_dataset = SMILESDataset({key: val[test_idx] for key, val in encoded_data.items()})

class InputDebugCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        torch.cuda.empty_cache()  # Clear cache after each step
        #if state.global_step % 100 == 0:  # Example condition to check
            #print("Cleared CUDA cache")

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

def model_init():
    return AutoModelForMaskedLM.from_pretrained("DeepChem/ChemBERTa-5M-MLM")

def compute_metrics(p: EvalPrediction):
    preds = p.predictions.argmax(-1)
    labels = p.label_ids
    mask = labels != -100
    num_correct = (preds[mask] == labels[mask]).sum().item()
    num_total = mask.sum().item()
    accuracy = num_correct / num_total
    return {"accuracy": accuracy}

# Define the best hyperparameters from trial
best_hyperparameters = {
    'learning_rate': 4.249894798853819e-05,
    'num_train_epochs': 5,  # Change to 'None' or a large number to train until convergence
    'per_device_train_batch_size': 16,
    'weight_decay': 0.05704196058538424
}


training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    save_strategy='epoch',
    logging_dir='./logs',
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model='loss',
    learning_rate=best_hyperparameters["learning_rate"],
    num_train_epochs=best_hyperparameters["num_train_epochs"],  # Set a very high number to allow training until convergence
    per_device_train_batch_size=best_hyperparameters["per_device_train_batch_size"],
    weight_decay=best_hyperparameters["weight_decay"],
    fp16=True,  # Mixed precision training
    report_to="wandb"  # Report metrics to W&B
)


trainer = Trainer(
    model_init=model_init,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    callbacks=[InputDebugCallback()],
    compute_metrics=compute_metrics
)

# Accumulative evaluation
def accumulative_eval(trainer, eval_dataset, batch_size):
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size)
    total_loss, total_correct, total_count = 0, 0, 0
    model.eval()
    with torch.no_grad():
        for batch in eval_dataloader:
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs)
            logits = outputs.logits
            loss = outputs.loss
            labels = inputs['labels']
            mask = labels != -100
            preds = logits.argmax(-1)
            correct = (preds[mask] == labels[mask]).sum().item()
            count = mask.sum().item()
            total_loss += loss.item() * count
            total_correct += correct
            total_count += count
    avg_loss = total_loss / total_count
    accuracy = total_correct / total_count
    return {"eval_loss": avg_loss, "eval_accuracy": accuracy}

# Start training
trainer.train()

# Custom evaluation to handle large datasets
eval_results = accumulative_eval(trainer, val_dataset, best_hyperparameters["per_device_train_batch_size"])
print("Evaluation Results:", eval_results)
wandb.log(eval_results)

# Evaluate on test dataset
test_results = accumulative_eval(trainer, test_dataset, best_hyperparameters["per_device_train_batch_size"])
print("Test Results:", test_results)
wandb.log(test_results)

# Finish the W&B run
wandb.finish()

# Save the trained model and tokenizer
model.save_pretrained("./trained_chemberta")
tokenizer.save_pretrained("./trained_chemberta")