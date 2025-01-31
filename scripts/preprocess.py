import requests
import re
from sklearn.model_selection import train_test_split
import torch

# Download Shakespeare's Sonnets from Project Gutenberg
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
response = requests.get(url)
text = response.text

# Save the raw text to a file
with open("/home/itachi/Mini-GPT/data/raw/shakespeare_sonnets.txt", "w", encoding="utf-8") as f:
    f.write(text)


# Clean the text
def clean_text(text):
    # Remove all non-ASCII characters
    text = re.sub(r'[^\x00-\x7f]',r'', text)
    text = "<start> " + text + " <end>"       # Add start and end tokens
    
    return text

# Load and clean the text
with open("/home/itachi/Mini-GPT/data/raw/shakespeare_sonnets.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
cleaned_text = clean_text(raw_text)

# Save cleaned text
with open("/home/itachi/Mini-GPT/data/processed/cleaned_sonnets.txt", "w", encoding="utf-8") as f:
    f.write(cleaned_text)


# Create vocabulary and tokenizer
vocab = sorted(set(cleaned_text))
vocab_size = len(vocab)
char_to_idx = {char: idx for idx, char in enumerate(vocab)}
idx_to_char = {idx: char for char, idx in char_to_idx.items()}

# Tokenize the text
def tokenize(text):
    return [char_to_idx[char] for char in text]

tokenized_text = tokenize(cleaned_text)
# Save tokenized text and vocabulary
import json

with open("/home/itachi/Mini-GPT/data/processed/tokenized_sonnets.json", "w") as f:
    json.dump(tokenized_text, f)

with open("/home/itachi/Mini-GPT/data/processed/vocab.json", "w") as f:
    json.dump({"char_to_idx": char_to_idx, "idx_to_char": idx_to_char}, f)



# Split the tokenized text
train_data, val_data = train_test_split(tokenized_text, test_size=0.2, shuffle=False)

# Save the splits
with open("/home/itachi/Mini-GPT/data/processed/train_data.json", "w") as f:
    json.dump(train_data, f)

with open("/home/itachi/Mini-GPT/data/processed/val_data.json", "w") as f:
    json.dump(val_data, f)


def create_sequences(tokenized_text, seq_length):
    inputs, targets = [], []
    for i in range(len(tokenized_text) - seq_length):
        inputs.append(tokenized_text[i:i+seq_length])
        targets.append(tokenized_text[i+1:i+seq_length+1])
    return inputs, targets

# Define sequence length
seq_length = 64  # Adjust based on your dataset and memory constraints

# Create sequences for training and validation
train_inputs, train_targets = create_sequences(train_data, seq_length)
val_inputs, val_targets = create_sequences(val_data, seq_length)

# Save sequences
train_data = {
    "inputs": torch.tensor(train_inputs, dtype=torch.long),
    "targets": torch.tensor(train_targets, dtype=torch.long)
}
val_data = {
    "inputs": torch.tensor(val_inputs, dtype=torch.long),
    "targets": torch.tensor(val_targets, dtype=torch.long)
}

torch.save(train_data, "/home/itachi/Mini-GPT/data/processed/train_sequences.pt")
torch.save(val_data, "/home/itachi/Mini-GPT/data/processed/val_sequences.pt")