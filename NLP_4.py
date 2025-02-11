# import torch
# import torch.nn as nn
# import torch.optim as optim
# import math
# import torch.nn.functional as F
# from torchtext.datasets import WikiText2
# from torchtext.data.utils import get_tokenizer
# from torchtext.vocab import build_vocab_from_iterator
# from torch.utils.data import DataLoader, Dataset

# class PositionalEncoding(nn.Module):
#     def _init_(self, d_model, max_len=5000):
#         super(PositionalEncoding, self)._init_()
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         self.pe = pe.unsqueeze(0)
    
#     def forward(self, x):
#         return x + self.pe[:, :x.size(1)].to(x.device)

# class MultiHeadAttention(nn.Module):
#     def _init_(self, d_model, num_heads):
#         super(MultiHeadAttention, self)._init_()
#         assert d_model % num_heads == 0
#         self.d_k = d_model // num_heads
#         self.num_heads = num_heads
#         self.qkv_linear = nn.Linear(d_model, d_model * 3)
#         self.out_linear = nn.Linear(d_model, d_model)
#         self.softmax = nn.Softmax(dim=-1)
    
#     def forward(self, x, mask=None):
#         batch_size, seq_length, d_model = x.shape
#         qkv = self.qkv_linear(x).reshape(batch_size, seq_length, self.num_heads, 3 * self.d_k).permute(2, 0, 1, 3)
#         q, k, v = qkv.chunk(3, dim=-1)
#         scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
#         if mask is not None:
#             scores = scores.masked_fill(mask == 0, float('-inf'))
#         attention = self.softmax(scores)
#         output = (attention @ v).transpose(1, 2).reshape(batch_size, seq_length, d_model)
#         return self.out_linear(output)

# class FeedForward(nn.Module):
#     def _init_(self, d_model, d_ff):
#         super(FeedForward, self)._init_()
#         self.fc1 = nn.Linear(d_model, d_ff)
#         self.fc2 = nn.Linear(d_ff, d_model)
#         self.relu = nn.ReLU()
    
#     def forward(self, x):
#         return self.fc2(self.relu(self.fc1(x)))

# class TransformerEncoderLayer(nn.Module):
#     def _init_(self, d_model, num_heads, d_ff, dropout=0.1):
#         super(TransformerEncoderLayer, self)._init_()
#         self.attn = MultiHeadAttention(d_model, num_heads)
#         self.ff = FeedForward(d_model, d_ff)
#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.dropout = nn.Dropout(dropout)
    
#     def forward(self, x, mask=None):
#         x = self.norm1(x + self.dropout(self.attn(x, mask)))
#         x = self.norm2(x + self.dropout(self.ff(x)))
#         return x

# class TransformerEncoder(nn.Module):
#     def _init_(self, num_layers, d_model, num_heads, d_ff, vocab_size, max_len):
#         super(TransformerEncoder, self)._init_()
#         self.embedding = nn.Embedding(vocab_size, d_model)
#         self.pos_encoding = PositionalEncoding(d_model, max_len)
#         self.layers = nn.ModuleList([TransformerEncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)])
#         self.norm = nn.LayerNorm(d_model)
#         self.fc_out = nn.Linear(d_model, vocab_size)
    
#     def forward(self, x, mask=None):
#         x = self.embedding(x) + self.pos_encoding(x)
#         for layer in self.layers:
#             x = layer(x, mask)
#         x = self.norm(x)
#         return self.fc_out(x)

# # Load real dataset
# train_iter = WikiText2(split='train')
# tokenizer = get_tokenizer("basic_english")

# # Build vocabulary
# vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=["<unk>"])
# vocab.set_default_index(vocab["<unk>"])

# # Prepare dataset
# class TextDataset(Dataset):
#     def _init_(self, data, vocab, tokenizer, max_len):
#         self.data = [torch.tensor(vocab(tokenizer(line)), dtype=torch.long) for line in data]
#         self.max_len = max_len
    
#     def _len_(self):
#         return len(self.data)
    
#     def _getitem_(self, idx):
#         x = self.data[idx][:self.max_len]
#         pad_len = self.max_len - x.size(0)
#         x = F.pad(x, (0, pad_len))
#         return x, x

# dataset = TextDataset(WikiText2(split='train'), vocab, tokenizer, max_len=100)
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# # Train Transformer
# vocab_size = len(vocab)
# d_model = 512
# num_heads = 8
# d_ff = 2048
# num_layers = 6
# max_len = 100

# encoder = TransformerEncoder(num_layers, d_model, num_heads, d_ff, vocab_size, max_len)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(encoder.parameters(), lr=0.001)

# # Training loop
# for epoch in range(5):
#     for x, targets in dataloader:
#         optimizer.zero_grad()
#         output = encoder(x)
#         loss = criterion(output.view(-1, vocab_size), targets.view(-1))
#         loss.backward()
#         optimizer.step()
#     print(f"Epoch {epoch + 1}, Loss: {loss.item()}")