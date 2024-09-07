import os
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, AutoModelForMaskedLM, TrainingArguments, Trainer
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse

# Set environment variable to handle memory issues
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Initialize the DDP environment
def init_ddp(local_rank):
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)
    world_size = dist.get_world_size()
    return local_rank, world_size

# Load tokenizer and model
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

# Custom Dataset Class
class CustomDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

# Creating the datasets with the custom dataset class
train_dataset = CustomDataset(encoded_train_data)
val_dataset = CustomDataset(encoded_val_data)
test_dataset = CustomDataset(encoded_test_data)

# Prepare function to set up the DataLoader with DistributedSampler
def prepare(rank, world_size, dataset, batch_size=32, pin_memory=False, num_workers=0):
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, sampler=sampler)
    return dataloader

# Define the training function
def train_model(rank, world_size, hyperparameters):
    local_rank, world_size = init_ddp(rank)
    model.to(local_rank)
    model_ddp = DDP(model, device_ids=[local_rank])

    # Data loaders
    train_dataloader = prepare(local_rank, world_size, train_dataset)
    val_dataloader = prepare(local_rank, world_size, val_dataset)

    # Example training loop
    for epoch in range(hyperparameters['num_train_epochs']):
        model_ddp.train()
        for batch in train_dataloader:
            inputs = {k: v.to(local_rank) for k, v in batch.items()}
            outputs = model_ddp(**inputs)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Optionally print the loss
            if batch % 50 == 0:
                print(f"Rank {local_rank}, Epoch {epoch}, Batch {batch}, Loss: {loss.item()}")

        # Validation step at the end of each epoch
        model_ddp.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                inputs = {k: v.to(local_rank) for k, v in batch.items()}
                outputs = model_ddp(**inputs)
                val_loss += outputs.loss.item()
        val_loss /= len(val_dataloader)
        print(f"Rank {local_rank}, Epoch {epoch}, Validation Loss: {val_loss}")
        
    # Evaluate on the test set after training
    test_dataset = CustomDataset(encoded_test_data)  # Assuming encoded_test_data is predefined
    test_dataloader = prepare(local_rank, world_size, test_dataset)
    model_ddp.eval()
    test_loss = 0
    with torch.no_grad():
        for batch in test_dataloader:
            inputs = {k: v.to(local_rank) for k, v in batch.items()}
            outputs = model_ddp(**inputs)
            test_loss += outputs.loss.item()
    test_loss /= len(test_dataloader)
    print(f"Test Loss: {test_loss}")

    # Clear CUDA memory
    torch.cuda.empty_cache()

    # Save the trained model and tokenizer
    if local_rank == 0:
        model.module.save_pretrained("./trained_chemberta_10perc_data")
        tokenizer.save_pretrained("./trained_chemberta_10perc_data")

    # Clean up distributed training setup
    cleanup()

def cleanup():
    dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()

    best_hyperparameters = {
        'learning_rate': 4.249894798853819e-05,
        'num_train_epochs': 5,
        'per_device_train_batch_size': 4,
        'weight_decay': 0.05704196058538424
    }

    train_model(args.local_rank, torch.cuda.device_count(), best_hyperparameters)




   


 



