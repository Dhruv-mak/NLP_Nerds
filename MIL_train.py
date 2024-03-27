import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
import json

def load_data(file_path):
    prompts = []
    labels = []
    i = 0
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            prompt_text = data['prompt']['text']
            toxicity = data['prompt']['toxicity']
            if toxicity is not None:
                prompts.append(prompt_text)
                labels.append([toxicity])
            else:
                i += 1
    print("none vals:", i)
    return prompts, torch.tensor(labels)

class TextDataset(Dataset):
    def __init__(self, prompts, labels, tokenizer, max_length):
        self.prompts = prompts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        label = self.labels[idx]
        inputs = self.tokenizer(prompt, return_tensors='pt', max_length=self.max_length, padding='max_length', truncation=True)
        input_ids = inputs['input_ids'].squeeze(0)
        attention_mask = inputs['attention_mask'].squeeze(0)
        return input_ids, attention_mask, label

class MILNetwork(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(MILNetwork, self).__init__()
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, h = self.gru(x)
        h = h.squeeze(0)
        out = self.fc(h)
        return self.sigmoid(out)

# Load data
file_path = 'realtoxicityprompts-data/prompts.jsonl'
prompts, labels = load_data(file_path)

# Tokenizer and language model
model_name = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
language_model = AutoModel.from_pretrained(model_name)

# Create a DataLoader for the training data
max_length = 50
train_dataset = TextDataset(prompts, labels, tokenizer, max_length)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

# Initialize the MILNetwork model
hidden_size = 768
output_size = 1
mil_model = MILNetwork(hidden_size, output_size)

# Define the loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(mil_model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    for input_ids, attention_mask, targets in train_loader:
        # Get embeddings from the language model
        with torch.no_grad():
            outputs = language_model(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = outputs.last_hidden_state

        # Forward pass through the MILNetwork
        predictions = mil_model(embeddings)
        loss = criterion(predictions, targets.float())

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

    # Save the model checkpoint
    torch.save({
        'epoch': epoch,
        'model_state_dict': mil_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss.item(),
    }, f'checkpoint_epoch_{epoch+1}.pth')

# Save the final trained model
torch.save(mil_model.state_dict(), 'mil_model.pth')
