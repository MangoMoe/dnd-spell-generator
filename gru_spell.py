# %% 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import string
import glob
import random
from tqdm import tqdm
import unidecode

# %% 
# TODO this will not be a seq-to-seq model, we are just learning the distribution
# for sequence to sequence it will be by letter
# This is the number of characters
print(len(string.printable))
# print(string.printable)
characters = list(string.printable)

# this length is used for both input size and output size

# Stealing this from the lab
# Turn string into list of longs
def char_tensor(string):
  tensor = torch.zeros(len(string)).long()
  for c in range(len(string)):
      tensor[c] = characters.index(string[c])
  return tensor

class GRU_Net(nn.Module):
    def __init__(self):
        super(GRU_Net, self).__init__()
        self.hidden_size = 200
        in_size = len(string.printable)
        out_size = len(string.printable)
        self.num_layers = 10
        # TODO idk how to set this up correctly
        self.embed = nn.Embedding(num_embeddings=in_size, embedding_dim=in_size)
        self.relu = nn.ReLU()
        self.gru = nn.GRU(
            input_size = in_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            # batch_first=True,
            )
        self.linear = nn.Linear(self.hidden_size, out_size)

    # TODO do I want to have the hidden state passed in here???
    def forward(self, in_char, hidden):
        emb = torch.unsqueeze(self.relu(self.embed(in_char)),0)
        out, hidden = self.gru(emb, hidden)
        ret = self.linear(out)
        return ret, hidden
    
    def init_hidden(self, num_chars):
        return torch.zeros(self.num_layers, num_chars, self.hidden_size).cuda()

    # Mostly copying the evaluate method from lab 6
    def sample_output(self, prime_str='A', predict_len=100, temperature=0.8):
        # we don't want to record gradients
        with torch.no_grad():
            hidden = self.init_hidden(len(prime_str)).cuda()
            out_str = "" + prime_str
            prime_tens = char_tensor(prime_str).cuda()
            out, hidden = self.forward(prime_tens, hidden)
            # TODO squeeze twice?
            out = torch.squeeze(torch.squeeze(out))
            print()
            # print(out.size())
            chars = torch.multinomial(torch.exp(out / temperature), predict_len)
            # print(chars.size())
            test = "" + prime_str
            for char in chars:
                test += characters[char]
            print(test)
            # print(chars)
            # test = ""
            # for thing in out:
            #     thing = torch.unsqueeze(thing, 0)
            #     print(thing)
            #     print(thing.size())
            #     char = torch.multinomial(torch.exp(thing / temperature), 1)
            #     print(char)
            #     test += characters[char]
            # print(test)


# %%
# construct the dataset
spells_list = []
for fil in glob.glob("spells/*.txt"):
    with open(fil) as spell_file:
        spells_list.append(spell_file.read())

file = unidecode.unidecode(open('./text_files/lotr.txt').read())
file_len = len(file)

# Copied this part from the lab just for sanity testing
chunk_len = 200
 
def random_chunk():
  start_index = random.randint(0, file_len - chunk_len)
  end_index = start_index + chunk_len + 1
  return file[start_index:end_index]
  
print(random_chunk())

def get_rand_spell():
    return random_chunk()
    # return random.choice(spells_list)

# %% 
# print(get_rand_spell())

# %% 
# Make the training loop here
n_epochs = 500
lr=1e-4

torch.cuda.empty_cache()

# Create model
model = GRU_Net()

model = model.cuda()
optim = torch.optim.Adam(model.parameters(), lr=lr)
model.train()
torch.cuda.empty_cache()

prog = tqdm(range(n_epochs))

# TODO TODO maybe do a bunch of batches instead of random selections
for i in prog:
    if i > n_epochs // 4 or i > n_epochs // 2:
        lr = lr / 10
    optim.zero_grad()
    spell = get_rand_spell()
    inp = char_tensor(spell[:-1]).cuda()
    target = char_tensor(spell[1:]).cuda()
    # target = torch.unsqueeze(target,0)
    hidden = model.init_hidden(len(inp))
    output, hidden = model(inp, hidden)
    # hidden = torch.squeeze(hidden,0)
    output = torch.squeeze(torch.squeeze(output, 1),0)
    if i % 500 == 0:
        print()
        print(output.size())
        # print(hidden.size())
        print(target.size())
        # print(output - target)
    loss = F.cross_entropy(output, target)
    prog.set_description("Loss: {:.4f}".format(loss.item()))
    loss.backward()
    optim.step()

model.sample_output()

# %% 
model.sample_output()

# %%
# Save and load model
file_name = "PROBABLY_BAD_TEMP_model_save.ckpt"
torch.save(model.state_dict(), file_name)

model = GRU_Net()
model.load_state_dict(torch.load(file_name))
model = model.cuda()
# model.eval()
