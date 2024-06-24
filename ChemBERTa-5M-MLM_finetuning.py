import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForMaskedLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling, EvalPrediction
import math
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# Step 2: Load Tokenizer and Model
tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-5M-MLM")
model = AutoModelForMaskedLM.from_pretrained("DeepChem/ChemBERTa-5M-MLM")

# Step 4: Move Model to CUDA (if necessary)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"Using device: {device}")

# Load and prepare data
with open('./Datasets/combined_nps.txt', 'r') as file:
    data = file.readlines()
    data = [line.strip() for line in data]

# Convert list to DataFrame to use sample method
data_df = pd.DataFrame(data, columns=['smiles'])
data_df = data_df.sample(frac=0.5, random_state=42)  # Sampling a fraction for demonstration

# Convert DataFrame back to list after sampling
data = data_df['smiles'].tolist()

# Split the data into training, validation, and testing sets
train_data, temp_data = train_test_split(data, test_size=0.2, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

def encode_smiles(smiles_list):
    return tokenizer(smiles_list, add_special_tokens=True, truncation=True, max_length=512, padding="max_length", return_tensors="pt")

# Encode each data set
encoded_train_data = encode_smiles(train_data)
encoded_val_data = encode_smiles(val_data)
encoded_test_data = encode_smiles(test_data)

# Debugging print
# print("Encoded Data Sample:", encoded_train_data.keys(), {k: v.shape for k, v in encoded_train_data.items()})

# since DataCollatorForLanguageModeling expects dictionaries
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        return {key: self.encodings[key][idx] for key in self.encodings}

# Creating the datasets with the custom dataset class
train_dataset = CustomDataset(encoded_train_data)
val_dataset = CustomDataset(encoded_val_data)
test_dataset = CustomDataset(encoded_test_data)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)


def compute_metrics(p: EvalPrediction):
    # Convert predictions and labels to tensors if they are numpy arrays
    if isinstance(p.predictions, np.ndarray):
        p.predictions = torch.tensor(p.predictions, device=model.device)
    if isinstance(p.label_ids, np.ndarray):
        p.label_ids = torch.tensor(p.label_ids, dtype=torch.long, device=model.device)

    preds = p.predictions.argmax(-1)
    labels = p.label_ids
    mask = labels != -100

    num_correct = (preds[mask] == labels[mask]).sum().item()
    num_total = mask.sum().item()
    accuracy = num_correct / num_total

    # Ensure predictions are reshaped correctly for the loss calculation
    loss = torch.nn.CrossEntropyLoss()(p.predictions.view(-1, model.config.vocab_size), p.label_ids.view(-1))
    perplexity = math.exp(loss.item())

    return {"accuracy": accuracy, "loss": loss.item(), "perplexity": perplexity}

training_args = TrainingArguments(
    output_dir='./results',              # Directory for saving output files
    evaluation_strategy='epoch',         # Evaluation is done at the end of each epoch
    save_strategy='epoch',               # Save model checkpoint at the end of each epoch
    logging_dir='./logs',                # Directory for storing logs
    logging_steps=10,                    # Log every 10 steps
    learning_rate=4.249894798853819e-05, # Learning rate from the hyperparameter optimization
    per_device_train_batch_size=16,      # Batch size from the hyperparameter optimization
    per_device_eval_batch_size=16,
    weight_decay=0.05704196058538424,    # Weight decay from the hyperparameter optimization
    num_train_epochs=20,                  # Number of training epochs from the hyperparameter optimization
    report_to=None                       # Disable external reporting to keep training local
)

# Define a function to print metrics at the end of each epoch
def print_and_save_metrics(metrics, filename="training_metrics.txt"):
    with open(filename, "a") as file:
        print(metrics, file=file)
    print(metrics)

class MyTrainer(Trainer):
    def on_epoch_end(self):
        super().on_epoch_end()
        output = self.evaluate()
        print_and_save_metrics(output)
        torch.cuda.empty_cache()

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        with torch.no_grad():  # Disable gradient computation
            output = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
            torch.cuda.empty_cache()
        return output
    
    def predict(self, test_dataset):
        with torch.no_grad():  # Disable gradient computation
            predictions, label_ids, metrics = super().predict(test_dataset)
            print_and_save_metrics(metrics, filename="final_test_metrics.txt")
            torch.cuda.empty_cache()
        return predictions, label_ids, metrics

trainer = MyTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)


# Start training
trainer.train()

# Clear CUDA memory
torch.cuda.empty_cache()

# Evaluate on the test set after training
trainer.predict(test_dataset)

# Save the trained model and tokenizer
model.save_pretrained("./trained_chemberta_half_data")
tokenizer.save_pretrained("./trained_chemberta_half_data")

# When usin DataParallelism
# model.module.save_pretrained("./trained_chemberta_half_data")

