import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch.multiprocessing as mp
import pandas as pd
from sklearn.model_selection import train_test_split
import logging

# Enhanced logging setup
logging.basicConfig(level=logging.DEBUG, filename='training_log.log', filemode='w',
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Environment setup for DDP
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12345'  # Updated port number
os.environ['CUDA_VISIBLE_DEVICES'] = '6, 7'

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    logging.info(f"Initialized DDP on rank {rank} in world of size {world_size}")

def cleanup():
    dist.destroy_process_group()
    logging.info("Destroyed process group")

class CustomDataset(Dataset):
    def __init__(self, tokenized_data):
        self.input_ids = tokenized_data['input_ids']
        self.attention_mask = tokenized_data['attention_mask']
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.input_ids[idx]  # Labels are the same as input_ids for Masked Language Modeling
        }

def load_and_prepare_data(filepath, sample_fraction=0.01):
    logging.info("Loading and sampling data.")
    with open(filepath, 'r') as file:
        data = file.readlines()
    data = [line.strip() for line in data]
    data_df = pd.DataFrame(data, columns=['smiles'])
    data_df = data_df.sample(frac=sample_fraction, random_state=42)
    logging.info(f"Data sampled: {len(data_df)} entries")
    return data_df['smiles'].tolist()

def setup_tokenizer_and_model(rank):
    logging.info("Loading tokenizer and model.")
    tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-5M-MLM")
    model = AutoModelForMaskedLM.from_pretrained("DeepChem/ChemBERTa-5M-MLM")
    model.to(rank)  # Move model to the correct device
    model = DDP(model, device_ids=[rank])  # Wrap model with DistributedDataParallel
    return tokenizer, model

def tokenize_data(tokenizer, data):
    if not data:
        logging.error("Received an empty list for tokenization")
    tokenized_output = tokenizer(data, truncation=True, padding="max_length", max_length=512, return_tensors="pt")
    if 'input_ids' not in tokenized_output or not tokenized_output['input_ids']:
        logging.error("No input_ids generated after tokenization")
    return tokenized_output

def evaluate(model, dataloader, device):
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0
    with torch.no_grad():
        for batch in dataloader:
            inputs = {key: val.to(device) for key, val in batch.items()}
            outputs = model(**inputs)
            loss = outputs.loss
            predictions = outputs.logits.argmax(dim=-1)
            total_loss += loss.item()
            total_correct += (predictions == inputs['labels']).sum().item()
            total_samples += inputs['input_ids'].size(0)
    
    accuracy = total_correct / total_samples if total_samples > 0 else 0
    average_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
    logging.info(f"Evaluation completed. Loss: {average_loss}, Accuracy: {accuracy}")
    return average_loss, accuracy

def train(model, dataloader, optimizer, device):
    model.train()  # Set the model to training mode
    total_loss = 0
    for batch in dataloader:
        inputs = {key: val.to(device) for key, val in batch.items()}
        optimizer.zero_grad()  # Clear any previously calculated gradients
        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()  # Compute gradient of the loss w.r.t. all trainable parameters
        optimizer.step()  # Update parameters
        total_loss += loss.item()
    average_loss = total_loss / len(dataloader)
    logging.info(f"Training completed for one epoch. Average Loss: {average_loss}")
    return average_loss

def main(rank, world_size):
    setup(rank, world_size)
    data_path = './Datasets/combined_nps.txt'  # Adjust path as necessary
    data = load_and_prepare_data(data_path)
    tokenizer, model = setup_tokenizer_and_model(rank)

    # Split data
    train_data, temp_data = train_test_split(data, test_size=0.2, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

    # Tokenization for datasets
    tokenized_train = tokenize_data(tokenizer, train_data)
    tokenized_val = tokenize_data(tokenizer, val_data)
    tokenized_test = tokenize_data(tokenizer, test_data)

    # DataLoader setup
    train_dataset = CustomDataset(tokenized_train)
    val_dataset = CustomDataset(tokenized_val)
    test_dataset = CustomDataset(tokenized_test)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    # Training loop
    num_epochs = 3
    for epoch in range(num_epochs):
        logging.info(f"Starting epoch {epoch+1}/{num_epochs}")
        train_loss = train(model, train_loader, optimizer, rank)

    # Evaluation
    val_loss, val_accuracy = evaluate(model, val_loader, rank)
    test_loss, test_accuracy = evaluate(model, test_loader, rank)

    # Log evaluation results
    logging.info(f"Validation Results - Loss: {val_loss}, Accuracy: {val_accuracy}")
    logging.info(f"Test Results - Loss: {test_loss}, Accuracy: {test_accuracy}")

    cleanup()

if __name__ == "__main__":
    world_size = 2  # Explicitly setting world size to 2
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
    logging.info("Process started with 2 devices.")
