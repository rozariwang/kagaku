import os
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForMaskedLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling, EvalPrediction, TrainerCallback
import wandb
import optuna
from optuna.integration import WandbCallback
from peft import LoraConfig, get_peft_model

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
data = data.sample(frac=0.25, random_state=42) 
smiles_list = data['smiles'].tolist()
encoded_data = encode_smiles(smiles_list)

# Split data into train and validation sets (90/10)
train_idx, val_idx = train_test_split(range(len(encoded_data['input_ids'])), test_size=0.1, random_state=42)
# Load the test data 
test_data_df = pd.read_csv('./hhwang/kagaku/Datasets/test.txt', header=None, names=['smiles'])
test_smiles_list = test_data_df['smiles'].tolist()
test_encoded_data = encode_smiles(test_smiles_list)

train_dataset = SMILESDataset({key: val[train_idx] for key, val in encoded_data.items()})
val_dataset = SMILESDataset({key: val[val_idx] for key, val in encoded_data.items()})
test_dataset = SMILESDataset(test_encoded_data)

class InputDebugCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        torch.cuda.empty_cache()  # Clear cache after each step

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

def compute_metrics(p: EvalPrediction):
    preds = p.predictions
    labels = p.label_ids
    mask = labels != -100
    num_correct = (preds[mask] == labels[mask]).sum().item()
    num_total = mask.sum().item()
    accuracy = num_correct / num_total
    return {"accuracy": accuracy}

def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids

class TrainEvalMetricsCallback(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        train_results = self.trainer.evaluate(eval_dataset=self.trainer.train_dataset)
        print(f"Train Results: {train_results}")
        wandb.log({"train_loss": train_results['eval_loss'], 
                   "train_accuracy": train_results['eval_accuracy']}, 
                  step=state.global_step)

        eval_results = self.trainer.evaluate(eval_dataset=self.trainer.eval_dataset)
        print(f"Eval Results: {eval_results}")
        wandb.log({"eval_loss": eval_results['eval_loss'], 
                   "eval_accuracy": eval_results['eval_accuracy']}, 
                  step=state.global_step)

# Hyperparameter optimization with Optuna
def objective(trial):
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 5e-4)
    num_train_epochs = trial.suggest_int('num_train_epochs', 1, 50)
    per_device_train_batch_size = trial.suggest_categorical('per_device_train_batch_size', [8, 16, 32])
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-5, 1e-1)
    lora_r = trial.suggest_int('lora_r', 4, 64)

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="masked_lm"
    )

    model_lora = get_peft_model(model, lora_config)
    model_lora.to(device)

    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy='epoch',
        save_strategy='epoch',
        logging_dir='./logs',
        logging_steps=100,
        load_best_model_at_end=False,
        metric_for_best_model='loss',
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=32,
        dataloader_num_workers=4,
        weight_decay=weight_decay,
        report_to="wandb",
        dataloader_pin_memory=True,
    )

    trainer = Trainer(
        model=model_lora,
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

    trainer.train()

    eval_results = trainer.evaluate(eval_dataset=val_dataset)
    return eval_results['eval_loss']

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=10, callbacks=[WandbCallback()])

# Train with the best hyperparameters
best_hyperparameters = study.best_params

lora_config = LoraConfig(
    r=best_hyperparameters['lora_r'],
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="masked_lm"
)

model_lora = get_peft_model(model, lora_config)
model_lora.to(device)

training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    save_strategy='epoch',
    logging_dir='./logs',
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model='loss',
    learning_rate=best_hyperparameters['learning_rate'],
    num_train_epochs=best_hyperparameters['num_train_epochs'],
    per_device_train_batch_size=best_hyperparameters['per_device_train_batch_size'],
    per_device_eval_batch_size=32,
    dataloader_num_workers=4,
    weight_decay=best_hyperparameters['weight_decay'],
    report_to="wandb",
    dataloader_pin_memory=True,
)

trainer = Trainer(
    model=model_lora,
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

trainer.train()

# Save the trained model and tokenizer explicitly at the end of training
final_checkpoint_dir = "./finetuned_25_MLM_chemberta"
os.makedirs(final_checkpoint_dir, exist_ok=True)
trainer.model.save_pretrained(final_checkpoint_dir)
tokenizer.save_pretrained(final_checkpoint_dir)
print(f"Final model saved to {final_checkpoint_dir}")

# Normal evaluation for the test dataset
test_results = trainer.evaluate(eval_dataset=test_dataset)
print("Test Results:", test_results)
wandb.log(test_results)

# Finish the W&B run
wandb.finish()
