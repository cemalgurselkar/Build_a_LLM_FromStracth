import torch
import torch.nn as nn
import torch.nn.functional as F
from Transformer import Transformer


# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
# ------------

torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data)-block_size,(batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train','val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X,Y = get_batch(split)
            logits, loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class GPTLanguageModel(nn.Module):
    def __init__(self,vocab_size,block_size,d_model,num_head,n_layer,dropout=0.1):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size,d_model)
        self.position_embedding_table = nn.Embedding(block_size,d_model)

        self.transformer = Transformer(
            src_vocab_size=vocab_size,
            tgt_vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_head,
            num_layers=n_layer,
            d_ff=4 * d_model,
            dropout=dropout
        )

        self.final_layer = nn.LayerNorm(d_model)
        self.head_lm = nn.Linear(d_model,vocab_size)
    
    def _init_weight(self,module):
        if isinstance(module,nn.Linear):
            torch.nn.init.normal_(module.weight,mean=0.0,std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module,nn.Embedding):
            torch.nn.init.normal_(module.weight,mean=0.0,std=0.02)
    
    def forward(self,idx,targets=None):
        
        B,T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T,device=device))
        x = tok_emb + pos_emb

        x = self.transformer(x,x)
        x = self.final_layer(x)
        logits = self.head_lm(x)

        loss = None
        if targets is not None:
            B,T,C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits,targets)
        
        return logits, loss
    
    def generate(self,idx,max_new_tokens,sampling=True):
        for _ in range(max_new_tokens):
            idx_cond = idx[:,-self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:,-1,:]
            probs = F.softmax(logits,dim=-1)
            if sampling:
                idx_next = torch.multinomial(probs,num_samples=1)
            else:
                idx_next = torch.argmax(probs,dim=-1,keepdim=True)
            idx = torch.cat((idx,idx_next),dim=1)
        return idx

model = GPTLanguageModel(
    vocab_size=vocab_size,
    block_size=block_size,
    d_model=n_embd,
    num_head=n_head,
    n_layer=n_layer,
    dropout=dropout)

model = model.to(device)

print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

optimizer = torch.optim.AdamW(model.parameters(),lr=learning_rate)


for iter in range(max_iters):

    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
    xb,yb = get_batch('train')
    logits, loss = model(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    context = torch.zeros((1,1),dtype=torch.long, device=device)
    print(decode(model.generate(context,max_new_tokens=500)[0].tolist()))