import torch
import torch.nn as nn
from torch.nn import functional as F

#hyperparams
batch_size=32
block_size=8
max_iters=3000
eval_interval=300
learning_rate=1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_inters=200
# ------------

torch.manual_seed(1337)

#read file
with open('input.txt','r',encoding='utf=8') as f:
    text = f.read()

#all the unique characters that occur in the text
chars = sorted(list(set(text)))
vocab_size = len(chars)
#create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s:[stoi[c] for c in s] #encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) #decoder: take a list of integers, output a string

#Let's now split up the data into train and validation sets
data = torch.tensor(encode(text),dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be training, rest validation
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    #generate a small batch of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size]for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_inters)
        for k in range(eval_inters):
            X, Y = get_batch(split)
            logits, loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class BigramLanguageModel(nn.Module):

    def __init__(self,vocab_size):
       super().__init__()
       #each token directly reads off the logits for the next token from a lookup table
       self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):

        #idx and targets are both (B,T) tensor of integers
        logits= self.token_embedding_table(idx) #(Batch = 4,Time = 8,Channels = vocab_size (65) )

        if targets is None:
            loss = None
        else:
        #logits are B x T x C entropy wants it in a B x C x T need to rearrange
            B, T, C =logits.shape
            logits = logits.view(B*T, C) 
         #we pass the function 2 dimenstions by combing B&T to make them long and satisfy C as 2nd dimension
            targets = targets.view(B*T)
        #targets need the same treatment so the model knows how to read the validation data properly
            loss = F.cross_entropy(logits, targets)

        #logits is scores for next character in the sequence
        return logits, loss

#we can evaluate fromt he model now we need to generate from the model

    def generate(self, idx, max_new_tokens):
        #idx is (B, T) array of indicies in the current context
        for _ in range(max_new_tokens):
            #get the predictions
            logits, loss = self(idx)
            #focus only on the last time step
            logits = logits[:, -1,:] #becomes (B, C)
            #apply softmax to get probabilites
            probs = F.softmax(logits, dim=-1) #(B,C)
            #sample from the distribution 
            idx_next = torch.multinomial(probs, num_samples=1)#(B, 1)
            #append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)#(B, T+1)
        return idx
    
model = BigramLanguageModel(vocab_size)
m = model.to(device)

optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))