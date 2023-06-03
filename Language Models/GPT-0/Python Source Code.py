import numpy as np 

import torch
import torch.nn as nn
from torch.nn import functional as F

stoi = {ch:i for i,ch in enumerate(sorted(list(set(text))))}
itos = {i:ch for i,ch in enumerate(sorted(list(set(text))))}

encoder = lambda s: [stoi[c] for c in s]
decoder = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(np.array(encoder(text)))
train = data[:int(0.9 * len(data))]
val = data[int(0.9 * len(data)):]

X = torch.stack([train[i : i + 8] for i in x])
y = torch.stack([train[i + 1: i + 8] for i in x])

def batch(dataset , batch_size = 4):
    
    data = train if dataset == "train" else val
    
    index = torch.randint(len(data) - block_size , 
                          (batch_size , ))
    
    X = torch.stack([data[i : i + block_size] for i in index])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in index])
    
    return X , y
X_batch , y_batch = batch("train")

embed_table = nn.Embedding(len(sorted(list(set(text)))) , 
                           len(sorted(list(set(text)))))
embed_table

vs = len((sorted(list(set(text)))))

class BigramLanguageModel(nn.Module):

    def __init__(self, vs):
        super().__init__()
        
        self.token_embedding_table = nn.Embedding(vs, vs)

    def forward(self, idx, targets=None):

        logits = self.token_embedding_table(idx)
        
        if targets is None:
            
            loss = None
        
        else:
            
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        
        for _ in range(max_new_tokens):
        
            logits, loss = self(idx)
            logits = logits[:, -1, :] 
        
            probs = F.softmax(logits, dim=-1) 
        
            idx_next = torch.multinomial(probs, num_samples=1) 
        
            idx = torch.cat((idx, idx_next), dim=1) 
        
        return idx

bg = BigramLanguageModel(vs)
logits, loss = bg(X_batch , y_batch)
optimizer = torch.optim.AdamW(bg.parameters() , lr = 1e-2)

bg.eval()
out = {}

def estimate_loss():
    
    out = {}
    bg.eval()
    
    for split in ['train', 'val']:
        
        losses = torch.zeros(200)
        
        for k in range(200):
            
            X, Y = batch(split)
            
            logits, loss = bg(X, Y)
            losses[k] = loss.item()
        
        out[split] = losses.mean()
    bg.train()
    return out
for iter in range(0):

    xb, yb = batch('train')

    logits, loss = bg(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    optimizer.step()
    
  context = torch.zeros((1, 1), dtype=torch.long, device=device)
  decoder(bg.generate(context, max_new_tokens=500)[0].tolist())
  
  key = nn.Linear(384 , 100 , bias = False)
query = nn.Linear(384 , 100 , bias = False)
value = nn.Linear(384 , 100 , bias = False)

class Head(nn.Module):

    def __init__(self, head_size):

        super().__init__()

        self.key = nn.Linear(384 , 100 , bias=False)
        self.query = nn.Linear(384 , 100 , bias=False)
        self.value = nn.Linear(384 , 100 , bias=False)

        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):

        B,T,C = x.shape

        k = self.key(x)   
        q = self.query(x) 
        
        wei = q @ k.transpose(-2 , -1) * k.shape[-1] ** -0.5 
        
        wei = wei.masked_fill(self.tril[:T , :T] == 0 , float('-inf')) 
        wei = F.softmax(wei, dim=-1) 

        wei = self.dropout(wei)
        
        v = self.value(x) 
        
        out = wei @ v 
        
        return out
      
  class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):

        super().__init__()

        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads , 384)

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        
        out = torch.cat([h(x) for h in self.heads] , dim=-1)
        out = self.dropout(self.proj(out))
        
        return out
 
class FeedFoward(nn.Module):

    def __init__(self):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * 384),
            nn.ReLU(),
            nn.Linear(4 * 384 , 384),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        
        return self.net(x)
      
    
class Block(nn.Module):

    def __init__(self, n_head):

        super().__init__()

        head_size = 384 // 10
        self.sa = MultiHeadAttention(10, 3)

        self.ffwd = FeedFoward(384)

        self.ln1 = nn.LayerNorm(384)
        self.ln2 = nn.LayerNorm(384)

    def forward(self, x):

        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))

        return x
vocab_size = len(sorted(list(set(text))))
 
class GPTLanguageModel(nn.Module):

    def __init__(self):
        
        super().__init__()
        
        self.token_embedding_table = nn.Embedding(vocab_size, 384)
        self.position_embedding_table = nn.Embedding(block_size, 384)
        
        self.blocks = nn.Sequential(*[Block(384, n_head = 5) for _ in range(5)])
        self.ln_f = nn.LayerNorm(384)
        
        self.lm_head = nn.Linear(384, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        
        if isinstance(module, nn.Linear):
            
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
            if module.bias is not None:
                
                torch.nn.init.zeros_(module.bias)
        
        elif isinstance(module, nn.Embedding):
            
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx) 
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) 
        
        x = tok_emb + pos_emb 
        x = self.blocks(x) 
        x = self.ln_f(x) 
        
        logits = self.lm_head(x) 

        if targets is None:
            
            loss = None
        else:
            
            B, T, C = logits.shape
            
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        
        for _ in range(max_new_tokens):
        
            idx_cond = idx[:, -block_size:]
        
            logits, loss = self(idx_cond)
        
            logits = logits[:, -1, :] 
        
            probs = F.softmax(logits, dim=-1) 
        
            idx_next = torch.multinomial(probs, num_samples=1) 
        
            idx = torch.cat((idx, idx_next), dim=1) 
        return idx

GPT = GPTLanguageModel()

for iter in range(0):

    xb, yb = batch('train')

    logits, loss = GPT(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decoder(GPT.generate(context, max_new_tokens=500)[0].tolist()))
open('more.txt', 'w').write(decoder(GPT.generate(context, max_new_tokens=10000)[0].tolist()))
