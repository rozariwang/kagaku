import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch.multiprocessing as mp
import torch.distributed as dist
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, filename='training_log.log', filemode='w',
                    format='%(asctime)s - %(levelname)s - %(message)s')

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'
os.environ['CUDA_VISIBLE_DEVICES'] = '6, 7'

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    logging.info(f"Initialized process group for rank {rank} in world of size {world_size}")

def cleanup():
    dist.destroy_process_group()
    logging.info("Destroyed process group")

def encode_smiles(smiles_list, tokenizer):
    if not smiles_list:
        logging.error("Empty smiles list provided to tokenizer")
    encodings = tokenizer(smiles_list, add_special_tokens=True, truncation=True, max_length=512, padding="max_length", return_tensors="pt")
    if not encodings['input_ids'].size(0):
        logging.error("No input_ids generated by tokenizer")
    return encodings

def print_and_save_metrics(metrics, filename="training_metrics.txt"):
    with open(filename, "a") as file:
        print(metrics, file=file)
    logging.info(f"Metrics saved and printed: {metrics}")

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
        logging.debug("CustomDataset initialized")

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        try:
            item = {key: val[idx] for key, val in self.encodings.items()}
            if not item['input_ids'].size(0):
                logging.warning(f"No input_ids for index {idx}")
            return item
        except IndexError as e:
            logging.error(f"IndexError: {str(e)} at index {idx}")
            raise

def evaluate(model, dataloader, device):
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0
    with torch.no_grad():
        for batch in dataloader:
            inputs = {'input_ids': batch['input_ids'].to(device), 'attention_mask': batch['attention_mask'].to(device)}
            outputs = model(**inputs)
            loss = outputs.loss
            total_loss += loss.item()
            predictions = outputs.logits.argmax(dim=-1)
            total_correct += (predictions == batch['labels'].to(device)).sum().item()
            total_samples += batch['input_ids'].size(0)

    accuracy = total_correct / total_samples
    logging.info(f"Eval Loss: {total_loss / len(dataloader)}, Accuracy: {accuracy}")
    return total_loss / len(dataloader), accuracy

def main(rank, world_size):
    try:
        setup(rank, world_size)

        tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-5M-MLM")
        model = AutoModelForMaskedLM.from_pretrained("DeepChem/ChemBERTa-5M-MLM")
        model = model.to(rank)
        model = DDP(model, device_ids=[rank])
        logging.info(f"Model loaded and wrapped with DDP on rank {rank}")

        with open('./Datasets/combined_nps.txt', 'r') as file:
            data = file.readlines()
        data = [line.strip() for line in data]
        logging.debug("Data loaded and processed")

        data_df = pd.DataFrame(data, columns=['smiles'])
        data_df = data_df.sample(frac=0.01, random_state=42)
        data = data_df['smiles'].tolist()

        train_data, temp_data = train_test_split(data, test_size=0.2, random_state=42)
        val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
        logging.info("Data split into training, validation, and test sets")

        encoded_train_data = encode_smiles(train_data, tokenizer)
        encoded_val_data = encode_smiles(val_data, tokenizer)
        encoded_test_data = encode_smiles(test_data, tokenizer)

        train_dataset = CustomDataset(encoded_train_data)
        val_dataset = CustomDataset(encoded_val_data)
        test_dataset = CustomDataset(encoded_test_data)

        train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        logging.info("DataLoaders setup completed")

        optimizer = torch.optim.Adam(model.parameters(), lr=4.249894798853819e-05)

        # Training loop
        model.train()
        for epoch in range(20):  # Number of epochs
            for batch in train_dataloader:
                inputs = {'input_ids': batch['input_ids'].to(rank), 'attention_mask': batch['attention_mask'].to(rank)}
                optimizer.zero_grad()
                outputs = model(**inputs)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                logging.info(f"Loss: {loss.item()}")

        # Evaluation
        eval_loss, eval_accuracy = evaluate(model, val_dataloader, rank)
        logging.info(f"Validation completed. Loss: {eval_loss}, Accuracy: {eval_accuracy}")

        # Test
        test_loss, test_accuracy = evaluate(model, test_dataloader, rank)
        logging.info(f"Test completed. Loss: {test_loss}, Accuracy: {test_accuracy}")

        logging.info("Training completed")

        # Cleanup and save model
        model.module.save_pretrained("./trained_chemberta_half_data")
        tokenizer.save_pretrained("./trained_chemberta_half_data")
        logging.info("Model and tokenizer saved successfully")
        cleanup()
    except Exception as e:
        logging.exception("An error occurred during training: {}".format(str(e)))
        cleanup()
        raise

if __name__ == "__main__":
    world_size = 2
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
    logging.info("Process started with 2 devices.")


