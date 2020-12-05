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
from matplotlib import pyplot as plt

# %% 
# TODO this will not be a seq-to-seq model, we are just learning the distribution
# for sequence to sequence it will be by letter
# This is the number of characters
print(len(string.printable))
# print(string.printable)
characters = list(string.printable)
print(len(characters))

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
        self.num_layers = 2
        # TODO idk how to set this up correctly
            # Maybe try different embedding sizes
        # TODO try old embedding size and see if it impacts it significantly
        self.embed = nn.Embedding(num_embeddings=in_size, embedding_dim=self.hidden_size)
        self.relu = nn.ReLU()
        # self.temp_linear = nn.Linear(in_size, self.hidden_size)
        # TODO try using the dropout parameter as well
        self.gru = nn.GRU(
            # input_size = in_size,
            input_size = self.hidden_size,
            hidden_size = self.hidden_size,
            num_layers = self.num_layers,
            )
        self.linear = nn.Linear(self.hidden_size, out_size)
        # self.soft_max = nn.Softmax()

    def forward(self, in_char, hidden):
        # emb = torch.unsqueeze(self.relu(self.embed(in_char)),0)
        emb = self.embed(in_char).view(1,1,-1)
        out, hidden = self.gru(emb, hidden)
        # out = self.relu(self.temp_linear(emb))
        # ret = self.linear(out)
        # print(hidden.size()) # 3,1,100
        # print(hidden[-1].size()) # 1,100
        # print(out.size()) # 1,1,100
        # print(out[-1].size()) # 1,100
        # ret = hidden[-1]
        # TODO why did the TA version have a relu at the end?
        # ret = self.relu(self.linear(out[-1]))
        ret = self.relu(self.linear(out))
        # print(ret.size())
        # ret = self.soft_max(ret)
        # print(out[-1] - hidden[-1]) # These are the same -_-
        return ret, hidden
    
    # def init_hidden(self, num_chars):
    #     return torch.zeros(self.num_layers, num_chars, self.hidden_size).cuda()
    def init_hidden(self):
        return torch.zeros(self.num_layers, 1, self.hidden_size).cuda()

    # Mostly copying the evaluate method from lab 6
    def sample_output(self, prime_str='A', predict_len=100, temperature=0.8):
        # we don't want to record gradients
        with torch.no_grad():
            # hidden = self.init_hidden(len(prime_str)).cuda()
            hidden = self.init_hidden().cuda()
            out_str = "" + prime_str
            prime_tens = char_tensor(prime_str).cuda()
            out, hidden = self.forward(prime_tens, hidden)
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

file = unidecode.unidecode(open('./text_files/lotr.txt', encoding="utf8").read())
file_len = len(file)

# Copied this part from the lab just for sanity testing
chunk_len = 400
 
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
lr=1e-3
n_epochs = 10000

torch.cuda.empty_cache()

# Create model
# model = GRU_Net()
# losses = []

model = model.cuda()
optim = torch.optim.Adam(model.parameters(), lr=lr)
model.train()
torch.cuda.empty_cache()

prog = tqdm(range(n_epochs))
loss_fn = nn.CrossEntropyLoss()

# TODO TODO maybe do a bunch of batches instead of random selections
for i in prog:
    # TODO add this back once you have the basics working
    # if i > n_epochs // 4 or i > n_epochs // 2:
    if i > n_epochs - (n_epochs // 4):
        lr = lr / 10
    optim.zero_grad()

    spell = get_rand_spell()
    inp = char_tensor(spell[:-1]).cuda()
    target = char_tensor(spell[1:]).cuda()
    # print()
    # print(len(inp))

    # Try going one character at a time, since the other does not seem to be working
    loss = 0
    # hidden = model.init_hidden(1)
    hidden = model.init_hidden()
    for j in range(len(inp)):
        # print()
        # print("---------")
        # print(inp[j+1])
        # # print(target[j])
        # print()

        temp = torch.unsqueeze(inp[j], 0)
        output, hidden = model(temp, hidden)
        # output = torch.squeeze(output, 1)
        output = torch.squeeze(output, 0)
        # print("OTHER SIZE: {}".format(output.size()))
        # output = torch.unsqueeze(output,0)

        targ = torch.unsqueeze(target[j], 0)
        # targ = target[j]

        # print(targ.item())
        # print(output[0,targ])
        # print(targ.size())
        # print(output.size())
        # print(torch.sum(output))
        # loss += F.cross_entropy(output, targ)

        _loss = loss_fn(output, targ)
        # _loss = loss_fn(targ, output)
        # print("^^^^")
        # print(_loss)
        loss += _loss
    # print("+++++++++")
    # print(loss.item() / len(inp))
    losses.append(loss.item() / len(inp))
    # break

    # hidden = model.init_hidden(len(inp))
    # output, hidden = model(inp, hidden)
    # output = torch.squeeze(torch.squeeze(output, 1),0)
    # loss = F.cross_entropy(output, target)
    # losses.append(loss.item())

    loss.backward()
    optim.step()

    if i % 10 == 0:
        prog.set_description("Avg Loss: {:.4f}".format(np.mean(losses[:-10])))
        print()
        print(targ.item())
        print(output[0,targ].item())
        print(torch.max(output).item())
        # print(output)
        print()
        model.sample_output()
        print()

model.sample_output()

# %% 
# Make a loss graph
plt.plot(losses)
plt.show()
# TODO could it need something like batch normalization or like regularization or something?
#   or even just batches in general...

# %% 
model.sample_output()

# %%
# Save and load model
file_name = "TEMP_model_save.ckpt"
torch.save(model.state_dict(), file_name)

model = GRU_Net()
model.load_state_dict(torch.load(file_name))
model = model.cuda()
# model.eval()
