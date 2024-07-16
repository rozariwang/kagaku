import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling
from datasets import load_dataset
from torch.utils.data import DataLoader
import random
import time
import math

# Function to load the model and tokenizer
def load_model_and_tokenizer(model_path):
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # Print tokenizer configuration
    print(tokenizer)
    # Load the model
    model = AutoModelForMaskedLM.from_pretrained(model_path)
    # Print model configuration
    print(model.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model

# Function to preprocess data
def preprocess_data(tokenizer, file_path, num_samples=2000, seed=42):
    random.seed(seed)
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    if len(lines) > num_samples:
        lines = random.sample(lines, num_samples)

    # Tokenize the selected lines
    tokenized_data = tokenizer(lines, truncation=True, padding="max_length", max_length=512, return_tensors="pt")

    return tokenized_data

# Function to calculate PPPL
def calculate_pppl(tokenized_data, model, tokenizer):
    model.eval()  # Set the model to evaluation mode
    total_pll = 0  # Initialize total pseudo-log-likelihood to 0
    num_tokens = 0  # Initialize number of tokens to 0

    with torch.no_grad():  # Disable gradient calculation
        input_ids = tokenized_data['input_ids'].to(model.device)
        attention_mask = tokenized_data['attention_mask'].to(model.device)

        for i in range(input_ids.size(0)):  # Iterate over each sequence in the batch
            input_ids_i = input_ids[i]
            attention_mask_i = attention_mask[i]

            sentence_pll = 0  # Initialize sentence PLL

            for j in range(1, input_ids_i.size(0) - 1):  # Iterate over each token position, avoid first and last token
                if input_ids_i[j] == tokenizer.pad_token_id:
                    break  # Skip padding tokens

                masked_input_ids = input_ids_i.clone()
                masked_input_ids[j] = tokenizer.mask_token_id  # Mask the j-th token

                # Forward pass through the model
                outputs = model(input_ids=masked_input_ids.unsqueeze(0), attention_mask=attention_mask_i.unsqueeze(0))
                logits = outputs.logits

                # Calculate log probability of the original token at position j
                token_log_prob = torch.log_softmax(logits[0, j], dim=-1)[input_ids_i[j]].item()
                sentence_pll += token_log_prob  # Add to sentence PLL

            total_pll += sentence_pll
            num_tokens += torch.sum(attention_mask_i).item()  # Count valid tokens (non-padding)

    if num_tokens == 0:
        print("No valid tokens to compute PPPL.")
        return float('inf')

    avg_pll = total_pll / num_tokens  # Calculate average PLL

    # Add check to prevent taking the exponential of an undefined value
    if math.isnan(avg_pll) or avg_pll == float('inf'):
        print(f"Invalid average PLL: {avg_pll}")
        return float('inf')

    pppl = math.exp(-avg_pll)  # Calculate PPPL
    return pppl

# Main function
def main():
    model_paths = ["DeepChem/ChemBERTa-5M-MLM", "./finetuned_25_MLM_chemberta", "./finetuned_50_MLM_chemberta", "./finetuned_100_MLM_chemberta"]
    datasets = ["./Datasets/test.txt", "./Datasets/pubchem_1k_smiles.txt", "./Datasets/tox21_cleaned.txt", "./Datasets/zinc_cleaned.txt"]

    # Store the preprocessed data for each dataset
    preprocessed_data = {}

    # Preprocess each dataset once and store the tokenized data
    tokenizer, _ = load_model_and_tokenizer(model_paths[0])
    for data_path in datasets:
        tokenized_data = preprocess_data(tokenizer, data_path)
        preprocessed_data[data_path] = tokenized_data

    # Use the same preprocessed data for each model
    for model_path in model_paths:
        tokenizer, model = load_model_and_tokenizer(model_path)
        for data_path in datasets:
            tokenized_data = preprocessed_data[data_path]
            pppl = calculate_pppl(tokenized_data, model, tokenizer)
            print(f"Model: {model_path}, Dataset: {data_path}, PPPL: {pppl}")
            # Clear CUDA cache after each dataset processing
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()