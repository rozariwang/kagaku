import os

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer, AutoModelForMaskedLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling, EvalPrediction
import torch.multiprocessing as mp
import torch.distributed as dist
import math
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# Step 1: Set CUDA Visible Devices
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def encode_smiles(smiles_list, tokenizer):
    return tokenizer(smiles_list, add_special_tokens=True, truncation=True, max_length=512, padding="max_length", return_tensors="pt")

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        print(f"CustomDataset __getitem__ idx: {idx}, item keys: {item.keys()}")
        return item

def compute_metrics(p: EvalPrediction, model):
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

def main(rank, world_size):
    setup(rank, world_size)

    # Load Tokenizer and Model
    tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-5M-MLM")
    model = AutoModelForMaskedLM.from_pretrained("DeepChem/ChemBERTa-5M-MLM")

    # Wrap the Model with DistributedDataParallel
    model = model.to(rank)
    model = DDP(model, device_ids=[rank])

    # Load and prepare data
    with open('./Datasets/combined_nps.txt', 'r') as file:
        data = file.readlines()
        data = [line.strip() for line in data]

    data_df = pd.DataFrame(data, columns=['smiles'])
    data_df = data_df.sample(frac=0.5, random_state=42)
    data = data_df['smiles'].tolist()

    train_data, temp_data = train_test_split(data, test_size=0.2, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

    encoded_train_data = encode_smiles(train_data, tokenizer)
    encoded_val_data = encode_smiles(val_data, tokenizer)
    encoded_test_data = encode_smiles(test_data, tokenizer)

    print(f"Train data size: {len(train_data)}, Encoded train data shape: {encoded_train_data['input_ids'].shape}")
    print(f"Validation data size: {len(val_data)}, Encoded validation data shape: {encoded_val_data['input_ids'].shape}")
    print(f"Test data size: {len(test_data)}, Encoded test data shape: {encoded_test_data['input_ids'].shape}")

    train_dataset = CustomDataset(encoded_train_data)
    val_dataset = CustomDataset(encoded_val_data)
    test_dataset = CustomDataset(encoded_test_data)

    class DebugDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):
        def __call__(self, features):
            print(f"data_collator input features: {features}")
            batch = super().__call__(features)
            print(f"data_collator output batch: {batch}")
            return batch

    data_collator = DebugDataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy='epoch',
        save_strategy='epoch',
        logging_dir='./logs',
        logging_steps=10,
        learning_rate=4.249894798853819e-05,
        per_device_train_batch_size=16,
        weight_decay=0.05704196058538424,
        num_train_epochs=25,
        report_to=None,
        local_rank=rank
    )

    class MyTrainer(Trainer):
        def on_epoch_end(self):
            super().on_epoch_end()
            output = self.evaluate()
            print_and_save_metrics(output)
            torch.cuda.empty_cache()

        def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
            output = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
            torch.cuda.empty_cache()
            return output

        def predict(self, test_dataset):
            predictions, label_ids, metrics = super().predict(test_dataset)
            print_and_save_metrics(metrics, filename="final_test_metrics.txt")
            torch.cuda.empty_cache()
            return predictions, label_ids, metrics

    train_dataloader = DataLoader(train_dataset, batch_size=training_args.per_device_train_batch_size, collate_fn=data_collator)
    print(f"Train DataLoader length: {len(train_dataloader)}")

    trainer = MyTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=lambda p: compute_metrics(p, model)
    )

    trainer.train()
    torch.cuda.empty_cache()
    trainer.predict(test_dataset)
    model.module.save_pretrained("./trained_chemberta_half_data")
    tokenizer.save_pretrained("./trained_chemberta_half_data")

    cleanup()

if __name__ == "__main__":
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
