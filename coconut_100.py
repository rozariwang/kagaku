import os
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling, EvalPrediction, TrainerCallback
import wandb
import optuna
import plotly.graph_objects as vis

# Expandable memory segments configuration for PyTorch
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Set the W&B API key
os.environ['WANDB_API_KEY'] = 'ab94aac01d8489527f36831ac31eacae67c98286'

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

encoded_data = encode_smiles(data['smiles'].tolist())
dataset = SMILESDataset(encoded_data)

train_idx, val_idx = train_test_split(range(len(encoded_data['input_ids'])), test_size=0.5, random_state=42)

train_dataset = SMILESDataset({key: val[train_idx] for key, val in encoded_data.items()})
val_dataset = SMILESDataset({key: val[val_idx] for key, val in encoded_data.items()})

class InputDebugCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        torch.cuda.empty_cache()  # Clear cache after each step
        if state.global_step % 100 == 0:  # Example condition to check
            print("Cleared CUDA cache")

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

# Start training
trainer.train()

# Finish the W&B run
wandb.finish()

# Save the trained model and tokenizer
model.save_pretrained("./trained_chemberta")
tokenizer.save_pretrained("./trained_chemberta")