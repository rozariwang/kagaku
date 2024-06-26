import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer, AutoModelForMaskedLM
import pandas as pd
import logging
import torch.multiprocessing as mp

# Basic logging setup to console
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def setup(rank, world_size):
    # Set visible devices and initialize DDP
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    os.environ['CUDA_VISIBLE_DEVICES'] = '6, 7'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    logging.info(f"Initialized DDP on rank {rank} in world of size {world_size}")

def cleanup():
    dist.destroy_process_group()
    logging.info("Destroyed process group")

def load_and_prepare_data(filepath, sample_fraction=0.01):
    logging.info("Loading data from file.")
    with open(filepath, 'r') as file:
        data = file.readlines()
    data = [line.strip() for line in data if line.strip()]
    logging.info(f"Loaded {len(data)} entries.")

    # Sampling a smaller fraction for simplicity
    data_df = pd.DataFrame(data, columns=['smiles'])
    data_df = data_df.sample(frac=sample_fraction, random_state=42)
    logging.info(f"Sampled {len(data_df)} entries.")
    return data_df['smiles'].tolist()

def setup_tokenizer_and_model():
    logging.info("Loading tokenizer and model.")
    tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-5M-MLM")
    model = AutoModelForMaskedLM.from_pretrained("DeepChem/ChemBERTa-5M-MLM")
    return tokenizer, model

def tokenize_data(tokenizer, data):
    if not data:
        logging.error("Received an empty list for tokenization")
    tokenized_output = tokenizer(data, truncation=True, padding="max_length", max_length=512)
    if 'input_ids' not in tokenized_output or not tokenized_output['input_ids']:
        logging.error("No input_ids generated after tokenization")
    return tokenized_output

def direct_model_test(tokenizer, model):
    logging.info("Running direct model test.")
    test_input = ["Test input string", "Another test string"]
    tokenized_data = tokenize_data(tokenizer, test_input)
    input_ids = torch.tensor(tokenized_data['input_ids']).to(0)
    attention_mask = torch.tensor(tokenized_data['attention_mask']).to(0)
    
    model = model.to(0)
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    print("Direct test output logits shape:", outputs.logits.shape)

def main(rank, world_size):
    setup(rank, world_size)
    data_path = './Datasets/combined_nps.txt'  # Update path as necessary
    data = load_and_prepare_data(data_path)
    tokenizer, model = setup_tokenizer_and_model()

    # Run a direct model test
    if rank == 0:  # Run this test only on the first process
        direct_model_test(tokenizer, model)

    # Regular processing starts here
    sample_data = data[:10]  # Smaller subset to ensure quick processing
    tokenized_data = tokenize_data(tokenizer, sample_data)
    input_ids = torch.tensor(tokenized_data['input_ids']).to(rank)
    attention_mask = torch.tensor(tokenized_data['attention_mask']).to(rank)
    
    model = model.to(rank)
    model = DDP(model, device_ids=[rank])
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logging.info("Model forward pass completed on rank {rank}.")

    print(f"Output shape on rank {rank}:", outputs.logits.shape)
    cleanup()

if __name__ == "__main__":
    world_size = 2  # Explicitly setting world size to 2
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)


