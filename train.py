import torch
#use https://pytorch.org

#to train the simplest model
import torch.nn as nn
from torch.nn import functional as F 

#for reproducibility
torch.manual_seed(1337)

#change this to read data from website not just text file
#read the file to train the model
with open('input.txt', 'r', encoding='utf-8') as f:
    text=f.read()

#test if the model is reading the file properly
print("length of dataset in characters:" ,len(text))

#test if its the right file
print(text[:1000])

#start building the vocab
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print("vocabsize", vocab_size)

print('----')

#tokenizer areas for improvement Googles(sentencepiece) or openAI(tiktoken)
#create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s:[stoi[c] for c in s] #encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) #decoder: take a list of integers, output a string

#test encoder and decoder
print(encode("hi there"))
print(decode(encode("hi there")))

print("----")

#encode the input and store it into a torch.Tensor
data = torch.tensor(encode(text),dtype=torch.long)
#print(data.shape, data.dtype)
#print(data[:1000]) #the 1000 characters we looked at earlier will to the GPT look like this

#Let's now split up the data into train and validation sets
n = int(0.9*len(data)) # first 90% will be training, rest validation
train_data = data[:n]
val_data = data[n:]



#this chunck of code can be removed
#set length of 8 for training 
block_size = 8
train_data[:block_size+1]

#code to train the transformer
x = train_data[:block_size]
y = train_data[1:block_size+1]
for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print(f"when input is {context} the target is: {target}")
# till here


#create batches of sample for parallel processing
batch_size = 4 #how many independent seqences will we process in parallel
block_size = 8 #what is the max context length for predictions

def get_batch(split):
    #generate a small batch of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size]for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

xb, yb = get_batch('train')
print('inputs:')
print(xb.shape)
print(xb)
print('targets:')
print(yb.shape)
print(yb)

print('----')

for b in range(batch_size): #batch dimension
    for t in range(block_size): #time dimension
        context = xb[b, :t+1]
        target = yb[b,t]
        print(f"when input is {context.tolist()} the target: {target}")

print('----')

print(xb) #our input to the transformer


# AI model to be trained bigram language model simplest one to implement
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

print("----")
m = BigramLanguageModel(vocab_size)
logits, loss =m(xb, yb)
print(logits.shape)
print(loss)

print("----")
print(decode(m.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist()))

#create a PyTorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

batch_size=32
for steps in range(10000):

    #sample a batch of data
    xb, yb = get_batch('train')

    #evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())