# %%
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm
from matplotlib import pyplot as plt
# TODO maybe you will have to use the permuted one, double check
from positional_encodings.positional_encodings import PositionalEncoding1D
from torchtext import data, datasets
import torchtext
import spacy
from utils_spell import load_spells_list

# %%
#   # TODO I'm not really sure if the encoders or decoders have the embeddings and positional encodings, I guess we'll see
# TODO we might also need masks or something similar, not sure
# Okay so it looks like pytorch does have a transformer architecture built in,
#   but it has both the encoder and decoder together and requires a target for 
#   the forward function. Luckily it also has nn.TransformerEncoder and 
#   nn.TransformerDecoder classes as well as a multi-head attention if we want that separate as well.
# TODO See the lab, it has weird embeddings and crap

# %%
class CustomTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=512):
        super(CustomTransformer, self).__init__()
        # TODO since both vocabularies should be the same, do we need two embeddings?
        # vocab_size should be a length
        self.input_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        # TODO should the parameters be reversed?
        self.output_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)

        # TODO starting with default parameters and changing as need be
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=1)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=3)

        # TODO I'm not sure what to set the channels parameter to
        self.positional_encoding = PositionalEncoding1D(channels=d_model)

        # TODO do we need to use xavier initialization on self.parameters() (see lab)?


    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # TODO take the max (i.e. infinity norm) of the memory sequence (and maybe some other norms) 
        #   from the encoder, to know how to generally generate your random seeds
        # TODO figure out how to use masks... https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        #   TODO use subsequent mask or square subsequent mask?
        # TODO why did the lab use deep copies for the positional encoding?

        # out_emb = self.output_embedding(tgt).cuda()
        # out_pos_enc = self.positional_encoding(out_emb).cuda()
        in_emb = self.input_embedding(src).cuda()
        print("src size: {}".format(src.size()))
        print("Embedding size: {}".format(in_emb.size()))
        # in_pos_enc = self.positional_encoding(in_emb).cuda()
        # trans_enc = self.transformer_encoder(in_pos_enc, src_mask).cuda()
        # res = self.transformer_decoder(out_pos_enc, trans_enc, src_mask, tgt_mask).cuda()
        # return res
        print("%" * 10)
        print(src.size())
        print(src_mask.size())
        print(tgt.size())
        print(tgt_mask.size())

        return self.transformer_decoder(
            self.positional_encoding(self.output_embedding(tgt)),
            self.transformer_encoder(
                self.positional_encoding(self.input_embedding(src)), 
                src_mask
                ), 
            src_mask,
            tgt_mask,
            )

    # TODO the default value for tgt should just be the start token
    def sample(self, tgt, memory):
        return self.transformer_decoder(tgt, memory)
    
    # TODO the default value for tgt should just be the start token
    def random_sample(self, tgt):
        # Generate a random memory sequence and plug it into self.transformer_decoder 
        #   instead of the output of the encoder
        # TODO
        random_mem = None
        return sample(target, random_mem)

# %%
# TODO See the data loading section of lab 7 for how to construct the tokenized vocabulary
# Borrowing heavily from the lab
nlp = spacy.load("en_core_web_sm")

print("Loading Text Dataset")

# Basically just extracts the plaintext of the tokens (objects) produced by the spacy tokenizer
# REMEMBER: this model is word-based not letter-based like GRU was
def tokenize(text):
    return [token.text for token in nlp.tokenizer(text)]

# Tokens for beginning of sentence, end of sentence, and blank (copied from lab)
BOS_TOKEN = '<s>'
EOS_TOKEN = '</s>'
BLANK_TOKEN = "<blank>"

# Um since I'm not translating to a new language, just trying to recreate the source vocab,
#   I'm just gonna use the same field for both, hopefully that works...
VOCAB_FIELD = data.Field(tokenize=tokenize, 
    init_token=BOS_TOKEN, 
    eos_token=EOS_TOKEN, 
    pad_token=BLANK_TOKEN,
    )

spells = load_spells_list()

fields = (["src", VOCAB_FIELD], ["trg", VOCAB_FIELD])
# the documentation for Example.fromlist isn't great so I'm mostly copying the other thing and hoping it works,
#   OH maybe fromlist really is referring to a string that is a list of tokens.
#   still it seems like we shouldn't need the tuple with spells[i] twice, TODO look into this
#   Should I have the spells offset by a character like in the GRU thing?
#       No the Batch class does that for us
examples = [torchtext.data.Example.fromlist((spells[i], spells[i]), fields) for i in range(len(spells))]

# TODO what's up with the MAX_LEN parameter in the lab???
# TODO I got rid of the val variable
train = data.Dataset(examples, fields=fields)

MIN_FREQ=1
# TODO I wonder if we will still need two fields... eh probably not
VOCAB_FIELD.build_vocab(train.src, min_freq=MIN_FREQ)


# %%
# copied these from the lab
global max_src_in_batch, max_tgt_in_batch

def batch_size_fn(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch,  len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch,  len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)

def generate_square_subsequent_mask(sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, src, trg=None, pad=0):
        self.src = src
        # okay so we need to try to do the square subsequent mask thing
        self.src_mask = (src != pad).unsqueeze(-2).transpose(0,2) # TODO this was the original

        # self.src_mask = generate_square_subsequent_mask(self.src.size(-1))
        # self.src_mask = torch.unsqueeze(self.src_mask, 0)

        # self.src_mask = (src != pad).unsqueeze(-2)
        # self.src_mask = self.make_std_mask(self.src, pad).transpose(0,2)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = \
                # generate_square_subsequent_mask(self.trg.size(-1))
                # self.make_std_mask(self.trg, pad)
                self.make_std_mask(self.trg, pad).transpose(0,2) # TODO original
            self.ntokens = (self.trg_y != pad).data.sum()
    
    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask

def rebatch(pad_idx, batch):
    "Fix order in torchtext to match ours"
    src, trg = batch.src.transpose(0, 1).cuda(), batch.trg.transpose(0, 1).cuda()
    return Batch(src, trg, pad_idx)

# Great I guess we need the data_iterator class too...
# TODO there is a lot of scaffolding code here gosh
class DataIterator(data.Iterator):
    def create_batches(self):
        if self.train:
            def pool(d, random_shuffler):
                for p in data.batch(d, self.batch_size * 100):
                    p_batch = data.batch(
                        sorted(p, key=self.sort_key),
                        self.batch_size, self.batch_size_fn)
                    for b in random_shuffler(list(p_batch)):
                        yield b
            self.batches = pool(self.data(), self.random_shuffler)
            
        else:
            self.batches = []
            for b in data.batch(self.data(), self.batch_size,
                                          self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))


# %%
# Get ready to train
pad_idx = VOCAB_FIELD.vocab.stoi[BLANK_TOKEN]
model = CustomTransformer(len(VOCAB_FIELD.vocab))
model = model.cuda()
loss_fn = nn.KLDivLoss(reduction='sum')
N_EPOCHS = 12
device = torch.device('cuda')
BATCH_SIZE = 1
lr = 5e-4

# I don't think we need a validation set for this
data_iter = DataIterator(train, batch_size=BATCH_SIZE, device=device, repeat=False, 
    sort_key=lambda x: (len(x.src), len(x.trg)), batch_size_fn=batch_size_fn, train=True)

optim = torch.optim.Adam(model.parameters(), lr=lr)

# %%
# Train
prog = tqdm(range(N_EPOCHS))
for epoch in prog:
    fixed_batch = (rebatch(pad_idx, b) for b in data_iter)
    total_tokens = 0
    total_loss = 0
    for i, batch in enumerate(fixed_batch):
        # batch.src = batch.src.cuda()
        # batch.trg = batch.trg.cuda()
        # batch.src_mask = batch.src_mask.cuda()
        # batch.trg_mask = batch.trg_mask.cuda()
        # TODO I might transpose the wrong thing here, so src might need to be transposed instead of src_mask
        # batch.src_mask = torch.squeeze(batch.src_mask,0)
        # batch.trg_mask = torch.squeeze(batch.trg_mask)
        # batch.src_mask = batch.src_mask.transpose(0,2)
        # batch.trg_mask = batch.trg_mask.transpose(0,2)
        batch.src = batch.src.transpose(0,1)
        print()
        print("-"*10)
        print(batch.src.size())
        print(batch.src_mask.size())
        print(batch.trg.size())
        print(batch.trg_mask.size())
        out = model(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
        loss = loss_fn(out, batch.trg_y, batch.ntokens)
        total_loss += loss.item()
        total_tokens += batch.ntokens
        loss.backward()
        optim.step()
        optim.zero_grad()
        if i % 50 == 0:
            prog.set_description("Batch: {}, Avg loss: {}".format(i, total_loss / total_tokens))

# %%
# TODO Other possibilities
#   look on dms guild or r/UnearthedArcana to increase your dataset
#   Maybe try label smoothing to improve performance

# %%
# Test from SO
layer=torch.nn.TransformerEncoderLayer(256, 8, 256, 0.1)
encoder=torch.nn.TransformerEncoder(layer, 6)
embed=torch.nn.Embedding(80000, 256)
src=torch.randint(0, 1000, (20, 95))
# print(src.size())
src = embed(src)
print(src.size())
# src_mask = torch.randint(0,2,(20, 20)).bool()
src_mask = torch.randint(0,2,(20, 20)).bool()
print(src_mask.size())
src_key_padding_mask = torch.randint(0,2,(95, 20)).bool()
print(src_key_padding_mask.size())
# output =  encoder(src, src_mask)
# output =  encoder(src, src_key_padding_mask=src_key_padding_mask)
output =  encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)





# %% 
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerModel(nn.Module):

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask):
        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output
# %%
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
# %%
import os
print(os.getcwd())
# %%
import io
import torch
from torchtext.utils import download_from_url, extract_archive
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

# url = 'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip'
# test_filepath, valid_filepath, train_filepath = extract_archive(download_from_url(url))
tokenizer = get_tokenizer('basic_english')
# TODO maybe you can use this function on your spells list
spells_text = load_spells_list()
vocab = build_vocab_from_iterator(map(tokenizer, spells_text))
# vocab = build_vocab_from_iterator(map(tokenizer,
#                                       iter(io.open(train_filepath,
#                                                    encoding="utf8"))))
print(vocab)

def data_process(raw_text_iter):
  data = [torch.tensor([vocab[token] for token in tokenizer(item)],
                       dtype=torch.long) for item in raw_text_iter]
  return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

# train_data = data_process(iter(io.open(train_filepath, encoding="utf8")))
# val_data = data_process(iter(io.open(valid_filepath, encoding="utf8")))
# test_data = data_process(iter(io.open(test_filepath, encoding="utf8")))
train_data = data_process(spells_text)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def batchify(data, bsz):
    # Divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

batch_size = 20
eval_batch_size = 10
train_data = batchify(train_data, batch_size)
# val_data = batchify(val_data, eval_batch_size)
# test_data = batchify(test_data, eval_batch_size)

# %%
bptt = 35
def get_batch(source, i):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape(-1)
    return data, target
# %%
ntokens = len(vocab.stoi) # the size of vocabulary
emsize = 200 # embedding dimension
nhid = 200 # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2 # the number of heads in the multiheadattention models
dropout = 0.2 # the dropout value
model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)

# %%
criterion = nn.CrossEntropyLoss()
# TODO these might need a look...
# lr = 5.0 # learning rate
lr = 1e-3 # learning rate
# optimizer = torch.optim.SGD(model.parameters(), lr=lr)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
scheduler = None

import time
def train():
    model.train() # Turn on the train mode
    total_loss = 0.
    start_time = time.time()
    src_mask = model.generate_square_subsequent_mask(bptt).to(device)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_batch(train_data, i)
        optimizer.zero_grad()
        if data.size(0) != bptt:
            src_mask = model.generate_square_subsequent_mask(data.size(0)).to(device)
        output = model(data, src_mask)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        log_interval = 200
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                    # epoch, batch, len(train_data) // bptt, scheduler.get_lr()[0],
                    epoch, batch, len(train_data) // bptt, lr,
                    elapsed * 1000 / log_interval,
                    cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

def evaluate(eval_model, data_source):
    eval_model.eval() # Turn on the evaluation mode
    total_loss = 0.
    src_mask = model.generate_square_subsequent_mask(bptt).to(device)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets = get_batch(data_source, i)
            if data.size(0) != bptt:
                src_mask = model.generate_square_subsequent_mask(data.size(0)).to(device)
            output = eval_model(data, src_mask)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
    return total_loss / (len(data_source) - 1)
# %%
best_val_loss = float("inf")
epochs = 20 # The number of epochs
best_model = None

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train()
    # val_loss = evaluate(model, val_data)
    val_loss = evaluate(model, train_data)
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
          'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                     val_loss, math.exp(val_loss)))
    print('-' * 89)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model

    if scheduler is not None:
        scheduler.step()
# %%
# TODO we never really gave it start, stop, and blank tokens, is this a problem?

def create_tensor_from_string(string, vocab):
    pre_tens = []
    words = string.split()
    for word in words:
        pre_tens.append(vocab.stoi[word])
    return torch.Tensor(pre_tens).long().to(device)

def decode_tensor_to_string(tens, vocab):
    string = ""
    for val in tens:
        # print(val.size())
        string += vocab.itos[val] + " "
    return string

# TODO this is a start, but now we need to make the freaking decoder and pass the output as memory I think
# TODO look at the `get_batch` function
model.eval()

def pick_next_word(start, vocab, model):
    num_words = len(start.split())
    start_tens = create_tensor_from_string(start, vocab)
    # print(start_tens)
    # print(decode_tensor_to_string(start_tens, vocab))
    src_mask = model.generate_square_subsequent_mask(len(start.split())).to(device)
    mask = (torch.ones(num_words, num_words) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)).to(device)
    # print("src_mask size: {}".format(src_mask.size()))
    # print(start_tens.size())
    out = model(start_tens, src_mask)
    # out = model(start_tens, mask)
    out = out.view(-1, ntokens)
    # print(out.size())
    # print(out.view(-1, ntokens).size())
    # TODO so I'm not really sure this output can be considered as log probabilities, its just an encoder? but maybe
    out = F.softmax(out, 1)
    out_ind = torch.max(out, 1)[1].long()
    # print(out_ind)
    # print("And finally:")
    out_strings = decode_tensor_to_string(out_ind[:len(start.split())], vocab)
    new_string = out_strings.split()[-1]
    # print(out_strings)
    # print(start +" "+ new_string)
    return new_string
    # print(decode_tensor_to_string(out_ind[:len(start.split())], vocab))

# start = "choose a creature within range"
start = "make a melee spell attack"
for i in range(10):
    start += " " + pick_next_word(start, vocab, model)

final = start
print(final)

# %%
