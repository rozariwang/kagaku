import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer, AutoModelForMaskedLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling, EvalPrediction
import torch.multiprocessing as mp
import torch.distributed as dist
import math
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
    logging.info(f"Initialized process group for rank {rank} in world of size {world_size}")

def cleanup():
    dist.destroy_process_group()
    logging.info("Destroyed process group")

def encode_smiles(smiles_list, tokenizer):
    logging.debug("Encoding smiles")
    return tokenizer(smiles_list, add_special_tokens=True, truncation=True, max_length=512, padding="max_length", return_tensors="pt")

def print_and_save_metrics(metrics, filename="training_metrics.txt"):
    with open(filename, "a") as file:
        print(metrics, file=file)
    logging.info(f"Metrics saved and printed: {metrics}")
    print(metrics)

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
        logging.debug("CustomDataset initialized")

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        logging.debug(f"CustomDataset __getitem__ idx: {idx}, item: {item}")
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

    metrics = {"accuracy": accuracy, "loss": loss.item(), "perplexity": perplexity}
    logging.info(f"Computed metrics: {metrics}")
    return metrics

class MyTrainer(Trainer):
    def on_epoch_end(self):
        super().on_epoch_end()
        output = self.evaluate()
        print_and_save_metrics(output)
        logging.info(f"Epoch ended with output: {output}")
        torch.cuda.empty_cache()

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        output = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        torch.cuda.empty_cache()
        logging.info(f"Evaluation completed with output: {output}")
        return output

    def predict(self, test_dataset):
        predictions, label_ids, metrics = super().predict(test_dataset)
        print_and_save_metrics(metrics, filename="final_test_metrics.txt")
        logging.info(f"Prediction completed with metrics: {metrics}")
        torch.cuda.empty_cache()
        return predictions, label_ids, metrics

def main(rank, world_size):
    logging.debug(f"Starting main function with rank {rank}")
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

    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy='epoch',
        save_strategy='epoch',
        logging_dir='./logs',
        logging_steps=10,
        learning_rate=4.249894798853819e-05,
        per_device_train_batch_size=16,
        weight_decay=0.05704196058538424,
        num_train_epochs=20,
        report_to=None,
        local_rank=rank
    )
    logging.info("Training arguments set")

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )
    logging.debug("Data collator configured")

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=training_args.per_device_train_batch_size, 
        collate_fn=data_collator,
        shuffle=True
    )
    logging.info("DataLoader setup completed")

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
    logging.info("Training completed")

    predictions, label_ids, metrics = trainer.predict(test_dataset)
    logging.info(f"Final test metrics: {metrics}")

    model.module.save_pretrained("./trained_chemberta_half_data")
    tokenizer.save_pretrained("./trained_chemberta_half_data")
    logging.info("Model and tokenizer saved successfully")

    cleanup()

if __name__ == "__main__":
    world_size = 2
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
    logging.info("Process started with 2 devices.")

