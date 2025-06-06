import torch
import torch.nn as nn
from transformers import DistilBertTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
import math

# Multi-Head Self-Attention Module
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)
        self.scale = torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
    
    def forward(self, x, mask=None):
        batch_size, seq_len, embed_dim = x.size()
        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        return self.out(out)

# Transformer Encoder Block
class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_hidden_dim, dropout=0.1):
        super(TransformerEncoderBlock, self).__init__()
        self.self_attention = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        attn_out = self.self_attention(x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x

# Transformer for Sentiment Analysis
class TransformerForSentiment(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, ff_hidden_dim, max_seq_len, dropout=0.1):
        super(TransformerForSentiment, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = self.create_pos_encoding(max_seq_len, embed_dim)
        
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, ff_hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, 2)  # 2 classes: negative, positive
    
    def create_pos_encoding(self, max_seq_len, embed_dim):
        pe = torch.zeros(max_seq_len, embed_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
    
    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.size()
        
        x = self.embedding(input_ids)
        pos_encoding = self.pos_encoding[:, :seq_len, :].to(x.device)
        x = x + pos_encoding
        x = self.dropout(x)
        
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(2)
        else:
            mask = None
        
        for layer in self.encoder_layers:
            x = layer(x, mask)
        
        pooled_output = x.mean(dim=1)
        logits = self.classifier(pooled_output)
        return logits

# Function to prepare dataset
def prepare_dataset(dataset, tokenizer, max_seq_len):
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            max_length=max_seq_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
    
    tokenized = dataset.map(tokenize_function, batched=True)
    tokenized.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    return tokenized

# Training function
def train_model(model, train_dataloader, device, num_epochs=3):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_dataloader:
            input_ids = batch['input_ids'].squeeze(1).to(device)
            attention_mask = batch['attention_mask'].squeeze(1).to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

# Sentiment analysis function
def analyze_sentiment(text, model, tokenizer, device, max_seq_len=50):
    encoding = tokenizer(
        text,
        max_length=max_seq_len,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probabilities = torch.softmax(logits, dim=-1)
        prediction = torch.argmax(probabilities, dim=-1).item()
    
    return "Positive" if prediction == 1 else "Negative", probabilities[0].cpu().numpy()

# Main function
def main():
    # Hyperparameters
    vocab_size = 30522
    embed_dim = 256
    num_heads = 8
    num_layers = 2
    ff_hidden_dim = 1024
    max_seq_len = 50
    batch_size = 16
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load tokenizer and dataset
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    dataset = load_dataset('imdb')
    train_dataset = prepare_dataset(dataset['train'], tokenizer, max_seq_len)
    
    # DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    model = TransformerForSentiment(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        ff_hidden_dim=ff_hidden_dim,
        max_seq_len=max_seq_len
    ).to(device)
    
    # Train the model
    print("Training model...")
    train_model(model, train_dataloader, device, num_epochs=10)
    
    # Test inference
    text = "I love this movie, it's absolutely fantastic!"
    sentiment, probs = analyze_sentiment(text, model, tokenizer, device, max_seq_len)
    print(f"\nText: {text}")
    print(f"Sentiment: {sentiment}")
    print(f"Probabilities (Negative, Positive): {probs}")

if __name__ == "__main__":
    main()