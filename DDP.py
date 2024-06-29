import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModelForMaskedLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling, EvalPrediction
import math
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import argparse

# Set environment variable to handle memory fragmentation and illegal memory access issues
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Initialize the DDP environment
def init_ddp(local_rank):
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)
    world_size = dist.get_world_size()
    return local_rank, world_size

# Load Tokenizer and Model
tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-5M-MLM")
model = AutoModelForMaskedLM.from_pretrained("DeepChem/ChemBERTa-5M-MLM")

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
class CustomDataset(Dataset):
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

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

# Prepare function to set up the DataLoader with DistributedSampler
def prepare(rank, world_size, dataset, batch_size=32, pin_memory=False, num_workers=0):
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, drop_last=False, shuffle=False, sampler=sampler, collate_fn=data_collator)
    return dataloader

# Custom Trainer Class with Debugging and DDP
class MyTrainer(Trainer):
    def train(self, resume_from_checkpoint=None, trial=None, **kwargs):
        # Get the train dataloader
        train_dataloader = self.get_train_dataloader()
        
        # Print input_ids before processing the first batch
        for batch in train_dataloader:
            print(f"First batch input_ids: {batch['input_ids']}")
            print(f"First batch attention_mask: {batch['attention_mask']}")
            break  # Only print the first batch
        
        # Continue with the normal training process
        super().train(resume_from_checkpoint, trial, **kwargs)
    
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

    def training_step(self, model, inputs):
        print(f"Inputs keys: {inputs.keys()}")  # Debugging: print keys of the inputs
        print(f"Sample input_ids: {inputs['input_ids'][0]}")
        print(f"Sample attention_mask: {inputs['attention_mask'][0]}")
        return super().training_step(model, inputs)

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--master_addr", type=str, default='127.0.0.1')
    parser.add_argument("--master_port", type=str, default='29500')
    args = parser.parse_args()
    
    # Set environment variables if not already set
    os.environ['RANK'] = str(args.local_rank)
    os.environ['WORLD_SIZE'] = str(args.world_size)
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port
    
    # Setup DDP
    local_rank, world_size = init_ddp(args.local_rank)

    # Global model is moved to GPU and wrapped with DDP
    global model
    model.to(local_rank)
    model = DDP(model, device_ids=[local_rank])

    # Prepare data loaders
    train_dataloader = prepare(local_rank, world_size, train_dataset, batch_size=16, pin_memory=True, num_workers=0)
    val_dataloader = prepare(local_rank, world_size, val_dataset, batch_size=16, pin_memory=True, num_workers=0)
    test_dataloader = prepare(local_rank, world_size, test_dataset, batch_size=16, pin_memory=True, num_workers=0)

    trainer = MyTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
        train_dataloader=train_dataloader,  # Assuming you add this to MyTrainer
        val_dataloader=val_dataloader 
    )

    # Start training
    trainer.train()

    # Clear CUDA memory
    torch.cuda.empty_cache()

    # Evaluate on the test set after training
    #trainer.predict(test_dataset)
    results = trainer.evaluate(test_dataloader)
    print(results)

    # Save the trained model and tokenizer
    if local_rank == 0:  # Save only on the main process
        model.module.save_pretrained("./trained_chemberta_10perc_data")
        tokenizer.save_pretrained("./trained_chemberta_10perc_data")

    # Cleanup
    dist.destroy_process_group()

if __name__ == "__main__":
    main()



