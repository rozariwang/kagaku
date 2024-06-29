import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForMaskedLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling, EvalPrediction
import math
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# Set environment variable to handle memory fragmentation and illegal memory access issues
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

# Load Tokenizer and Model
tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-5M-MLM")
model = AutoModelForMaskedLM.from_pretrained("DeepChem/ChemBERTa-5M-MLM")

# Wrap model with DataParallel
model = nn.DataParallel(model, device_ids=[6, 7])

print(f"Using DataParallel with devices: {model.device_ids}")

# Load and prepare data
with open('./Datasets/combined_nps.txt', 'r') as file:
    data = file.readlines()
    data = [line.strip() for line in data]

# Convert list to DataFrame to use sample method
data_df = pd.DataFrame(data, columns=['smiles'])
data_df = data_df.sample(frac=0.05, random_state=42)  # Sampling a fraction for demonstration

# Convert DataFrame back to list after sampling
data = data_df['smiles'].tolist()

# Split the data into training, validation, and testing sets
train_data, temp_data = train_test_split(data, test_size=0.2, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

# Ensure proper encoding of the data
def encode_smiles(smiles_list):
    return tokenizer(smiles_list, add_special_tokens=True, truncation=True, max_length=512, padding="max_length", return_tensors="pt")

# Encode each data set and ensure the inclusion of 'input_ids'
encoded_train_data = encode_smiles(train_data)
encoded_val_data = encode_smiles(val_data)
encoded_test_data = encode_smiles(test_data)

# Check if 'input_ids' are included in the encoded data
assert 'input_ids' in encoded_train_data, "Encoded train data does not include 'input_ids'"
assert 'input_ids' in encoded_val_data, "Encoded validation data does not include 'input_ids'"
assert 'input_ids' in encoded_test_data, "Encoded test data does not include 'input_ids'"

# Debug: Print keys and shapes of encoded data
print("Train data keys:", encoded_train_data.keys())
print("Validation data keys:", encoded_val_data.keys())
print("Test data keys:", encoded_test_data.keys())

# Debug: Print shapes of encoded data
print("Train data shapes:", {k: v.shape for k, v in encoded_train_data.items()})
print("Validation data shapes:", {k: v.shape for k, v in encoded_val_data.items()})
print("Test data shapes:", {k: v.shape for k, v in encoded_test_data.items()})

# Custom Dataset Class
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item


# Creating the datasets with the custom dataset class
train_dataset = CustomDataset(encoded_train_data)
val_dataset = CustomDataset(encoded_val_data)
test_dataset = CustomDataset(encoded_test_data)

# DataLoaders with pin_memory and num_workers, using data_collator
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True, collate_fn=data_collator)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True, collate_fn=data_collator)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True, collate_fn=data_collator)

# Function to print batch details for debugging
def debug_batch(batch):
    print("Batch keys:", batch.keys())
    for key, value in batch.items():
        print(f"Key: {key}, Shape: {value.shape}")

# Debug: Print details of the first batch in train_dataloader
print("Debugging first batch in train_dataloader:")
for batch in train_dataloader:
    debug_batch(batch)
    break  # Process only the first batch for debugging

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)


def compute_metrics(p: EvalPrediction):
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

    loss = torch.nn.CrossEntropyLoss()(p.predictions.view(-1, model.config.vocab_size), p.label_ids.view(-1))
    perplexity = math.exp(loss.item())

    return {"accuracy": accuracy, "loss": loss.item(), "perplexity": perplexity}

training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy='epoch',
    save_strategy='epoch',
    logging_dir='./logs',
    logging_steps=10,
    learning_rate=4.249894798853819e-05,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.05704196058538424,
    num_train_epochs=20,
    report_to=None
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

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        with torch.no_grad():
            output = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
            torch.cuda.empty_cache()
            print(torch.cuda.memory_summary())
        return output
    
    def predict(self, test_dataset):
        with torch.no_grad():
            predictions, label_ids, metrics = super().predict(test_dataset)
            print_and_save_metrics(metrics, filename="final_test_metrics.txt")
            torch.cuda.empty_cache()
            print(torch.cuda.memory_summary())
        return predictions, label_ids, metrics

trainer = MyTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataloader,
    eval_dataset=val_dataloader,
    compute_metrics=compute_metrics
)

# Start training
trainer.train()

# Clear CUDA memory
torch.cuda.empty_cache()

# Evaluate on the test set after training
trainer.predict(test_dataset)

# Save the trained model and tokenizer
model.module.save_pretrained("./trained_chemberta_10perc_data")
tokenizer.save_pretrained("./trained_chemberta_10perc_data")
