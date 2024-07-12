import os
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForMaskedLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling, EvalPrediction, TrainerCallback
import wandb

# Set max_split_size_mb to avoid memory fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
os.environ["WANDB_API_KEY"] = "ab94aac01d8489527f36831ac31eacae67c98286"

# Initialize W&B
wandb.init(project="chemberta-finetuning")

# Set device to GPU if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load ChemBERTa
tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-5M-MLM")
model = AutoModelForMaskedLM.from_pretrained("DeepChem/ChemBERTa-5M-MLM")

model.to(device)  # Move the model to the specified device

def encode_smiles(smiles_list):
    return tokenizer(smiles_list, add_special_tokens=True, truncation=True, max_length=512, padding="max_length", return_tensors="pt")

class SMILESDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        return {key: self.encodings[key][idx] for key in self.encodings}

data = pd.read_csv('./hhwang/kagaku/Datasets/train.txt', header=None, names=['smiles']) 
#data = pd.read_csv('./Datasets/train.txt', header=None, names=['smiles']) # for the interactive job
data = data.sample(frac=0.25, random_state=42) # for the interactive job
smiles_list = data['smiles'].tolist()
encoded_data = encode_smiles(smiles_list)

# Split data into train and validation sets (90/10)
train_idx, val_idx = train_test_split(range(len(encoded_data['input_ids'])), test_size=0.1, random_state=42)
# Load the test data 
test_data_df = pd.read_csv('./hhwang/kagaku/Datasets/test.txt', header=None, names=['smiles'])
#test_data_df = pd.read_csv('./Datasets/test.txt', header=None, names=['smiles']) # for the interactive job
#test_data_df = test_data_df.sample(frac=0.001, random_state=42) # for the interactive job & 50% & 25% data run
test_smiles_list = test_data_df['smiles'].tolist()
test_encoded_data = encode_smiles(test_smiles_list)

train_dataset = SMILESDataset({key: val[train_idx] for key, val in encoded_data.items()})
val_dataset = SMILESDataset({key: val[val_idx] for key, val in encoded_data.items()})
test_dataset = SMILESDataset(test_encoded_data)

class InputDebugCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        torch.cuda.empty_cache()  # Clear cache after each step
        #if state.global_step % 100 == 0:  # Example condition to check
            #print("Cleared CUDA cache")

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)


def compute_metrics(p: EvalPrediction):
    preds = p.predictions  # Directly use the predictions as they are the class indices
    labels = p.label_ids
    mask = labels != -100  # Ignore padding tokens or other special values
    num_correct = (preds[mask] == labels[mask]).sum().item()
    num_total = mask.sum().item()
    accuracy = num_correct / num_total
    return {"accuracy": accuracy}

### Look into what this function does ###
def preprocess_logits_for_metrics(logits, labels):
    """
    This function preprocesses logits by extracting them if in a tuple, and
    returns predicted class indices to save memory.
    """
    if isinstance(logits, tuple):
        logits = logits[0]  # Extract logits if they are in a tuple
    pred_ids = torch.argmax(logits, dim=-1)  # Compute the predicted class indices
    return pred_ids  # Return only the predictions

class TrainEvalMetricsCallback(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        # Compute train accuracy
        train_results = self.trainer.evaluate(eval_dataset=self.trainer.train_dataset)
        print(f"Train Results: {train_results}")
        wandb.log({"train_loss": train_results['eval_loss'], 
                   "train_accuracy": train_results['eval_accuracy']}, 
                  step=state.global_step)

        # Compute evaluation accuracy
        eval_results = self.trainer.evaluate(eval_dataset=self.trainer.eval_dataset)
        print(f"Eval Results: {eval_results}")
        wandb.log({"eval_loss": eval_results['eval_loss'], 
                   "eval_accuracy": eval_results['eval_accuracy']}, 
                  step=state.global_step)

# Define the best hyperparameters from trial
best_hyperparameters = {
    'learning_rate': 2.759070997751884e-05,
    'num_train_epochs': 25,  # Change to 'None' or a large number to train until convergence
    'per_device_train_batch_size': 16, #16
    'weight_decay': 0.06478239547856546
}


training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    save_strategy='epoch',
    logging_dir='./logs',
    logging_steps=100,
    load_best_model_at_end=False,
    metric_for_best_model='loss',
    learning_rate=best_hyperparameters["learning_rate"],
    num_train_epochs=best_hyperparameters["num_train_epochs"],  # Set a very high number to allow training until convergence
    per_device_train_batch_size=best_hyperparameters["per_device_train_batch_size"],
    per_device_eval_batch_size=32,  # Specify the batch size for evaluation
    dataloader_num_workers=4,  # Number of workers for data loading
    #gradient_accumulation_steps=4,  # Accumulate gradients over 4 steps
    weight_decay=best_hyperparameters["weight_decay"],
    #fp16=True,  # Mixed precision training
    report_to="wandb",  # Report metrics to W&B
    #eval_accumulation_steps=2,
    dataloader_pin_memory=True,
)


trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    callbacks=[InputDebugCallback()],
    compute_metrics=compute_metrics, 
    preprocess_logits_for_metrics=preprocess_logits_for_metrics
)

metrics_callback = TrainEvalMetricsCallback(trainer)
trainer.add_callback(metrics_callback)

# Start training
trainer.train()

# Normal evaluation for the test dataset
test_results = trainer.evaluate(eval_dataset=test_dataset)
print("Test Results:", test_results)
wandb.log(test_results)

#test
# Finish the W&B run
wandb.finish()

# Save the trained model and tokenizer explicitly at the end of training
final_checkpoint_dir = "./finetuned_25_MLM_chemberta"
os.makedirs(final_checkpoint_dir, exist_ok=True)
trainer.model.save_pretrained(final_checkpoint_dir)
tokenizer.save_pretrained(final_checkpoint_dir)
print(f"Final model saved to {final_checkpoint_dir}")