# %% 
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
from utils_spell import load_spells_list

import spacy
nlp = spacy.load("en_core_web_sm")

class TransformerModel(nn.Module):

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        print("Number of tokens was: {}".format(ntoken))
        print("Number of inputs was: {}".format(ninp))
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        # self.pos_encoder = PositionalEncoding1D(channels=ninp)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        # encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=1)
        # self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
        # TODO rename this to something else
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        # self.decoder = nn.Linear(ninp, ntoken)
        # TODO we might not have to use separate decoder and encoder modules after all... whelp
        decoder_layer = nn.TransformerDecoderLayer(d_model=ninp, nhead=nhead, dim_feedforward=nhid, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=nlayers)
        self.output_linear = nn.Linear(ninp, ntoken)

        self.init_weights()

    # TODO maybe try using nn.Transformer's built in funciton for this
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    # TODO why again is this necessary???
    # TODO maybe try different initializations on the actual transformer parts as well
    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        # self.decoder.bias.data.zero_()
        # self.decoder.weight.data.uniform_(-initrange, initrange)

    # TODO add target and target mask here
    def forward(self, src, src_mask, tgt, tgt_mask):
        # TODO not sure why the square root thing is necessary
        # src = self.encoder(src) * math.sqrt(self.ninp)
        # src = self.pos_encoder(src)
        # output = self.transformer_encoder(src, src_mask)
        # output = self.decoder(output)

        # TODO is the output of transformer_decoder already softmaxed or not???
        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src, src_mask)
        tgt = self.encoder(tgt) * math.sqrt(self.ninp)
        tgt = self.pos_encoder(tgt)
        output = self.transformer_decoder(tgt, memory, src_mask, tgt_mask)
        # Turn it into the right number of tokens
        output = self.output_linear(output)

        return output
# %%
# trying this one too, turns out it was causing the problem
# TODO maybe you will have to use the permuted one, double check
from positional_encodings.positional_encodings import PositionalEncoding1D

# TODO why the heck does the positional encoding have a dropout in it?
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
    # target = source[i+1:i+1+seq_len].reshape(-1)
    target = source[i+1:i+1+seq_len]
    # print("In get_batch")
    # print(data.size())
    # print(target.size())
    return data, target
# %%
ntokens = len(vocab.stoi) # the size of vocabulary
emsize = 200 # embedding dimension
nhid = 200 # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 5 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 5 # the number of heads in the multiheadattention models
dropout = 0.2 # the dropout value
model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)

# %%
# TODO TODO try label smoothing just copy it bro....
criterion = nn.CrossEntropyLoss()
# TODO these might need a look...

lr = 5.0 # learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

# optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# lr = 1e-4 # learning rate
# scheduler = None

import time
def train():
    model.train() # Turn on the train mode
    total_loss = 0.
    start_time = time.time()
    src_mask = model.generate_square_subsequent_mask(bptt).to(device)
    tgt_mask = model.generate_square_subsequent_mask(bptt).to(device)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        # TODO rename this to be singular
        data, targets = get_batch(train_data, i)
        optimizer.zero_grad()
        if data.size(0) != bptt:
            src_mask = model.generate_square_subsequent_mask(data.size(0)).to(device)
        # TODO check this
        if targets.size(0) != bptt:
            tgt_mask = model.generate_square_subsequent_mask(targets.size(0)).to(device)
        # print("%" * 10)
        # print(data.size())
        # print(src_mask.size())
        # print(targets.size())
        # print(tgt_mask.size())
        output = model(data, src_mask, targets, tgt_mask)
        # ????
        # output = torch.max(output, -1)[1]
        # print("Temp size: {}".format(temp.size()))
        # print(output.size())
        # loss = criterion(output.view(-1, ntokens), targets)
        # loss = criterion(output.view(-1, ntokens), targets.view(-1, ntokens))
        loss = criterion(output.view(-1, ntokens), targets.reshape(-1))
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
                  'loss {:5.2f} | custom {:10.2f}'.format(
                    # epoch, batch, len(train_data) // bptt, scheduler.get_lr()[0],
                    epoch, batch, len(train_data) // bptt, lr,
                    elapsed * 1000 / log_interval,
                    cur_loss, 1. / cur_loss))
            total_loss = 0
            start_time = time.time()

# TODO TODO TODO this one too...
def evaluate(eval_model, data_source):
    eval_model.eval() # Turn on the evaluation mode
    total_loss = 0.
    src_mask = model.generate_square_subsequent_mask(bptt).to(device)
    tgt_mask = model.generate_square_subsequent_mask(bptt).to(device)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets = get_batch(data_source, i)
            if data.size(0) != bptt:
                src_mask = model.generate_square_subsequent_mask(data.size(0)).to(device)
            if targets.size(0) != bptt:
                tgt_mask = model.generate_square_subsequent_mask(targets.size(0)).to(device)
            
            output = model(data, src_mask, targets, tgt_mask)
            output_flat = output.view(-1, ntokens)
            # loss = criterion(output.view(-1, ntokens), targets.reshape(-1))
            total_loss += len(data) * criterion(output_flat, targets.reshape(-1)).item()
    return total_loss / (len(data_source) - 1)
# %%
# Train
gc.collect()
torch.cuda.empty_cache()
best_val_loss = float("inf")
epochs = 10 # The number of epochs
best_model = None

for epoch in range(1, epochs + 1):
    gc.collect()
    torch.cuda.empty_cache()
    epoch_start_time = time.time()
    train()
    # val_loss = evaluate(model, val_data)
    gc.collect()
    torch.cuda.empty_cache()
    # TODO actually make a test/train split
    val_loss = evaluate(model, train_data)
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.8f} | '
          'valid ppl {:10.8f}'.format(epoch, (time.time() - epoch_start_time),
                                     val_loss, 1. / val_loss))
    print('-' * 89)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model

    if scheduler is not None:
        scheduler.step()

# %%
# TODO we never really gave it start, stop, and blank tokens, is this a problem?
def tokenize(text):
    return [token.text for token in nlp.tokenizer(text)]

def create_tensor_from_string(string, vocab):
    pre_tens = []
    # TODO use a tokenizer silly

    words = tokenize(string)
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
gc.collect()
torch.cuda.empty_cache()

def pick_next_word(start, vocab, model, temperature=0.8):
    num_words = len(start.split())
    start_tens = create_tensor_from_string(start, vocab)
    tgt_tens = torch.zeros(start_tens.size()).long().to(device)
    tgt_tens[:-1] = start_tens[1:]
    # print("&" * 10)
    # print(start_tens)
    # print(tgt_tens)
    # src, tgt = get_batch(start_tens, 0)
    # print(start_tens)
    # print(decode_tensor_to_string(start_tens, vocab))
    # src_mask = model.generate_square_subsequent_mask(len(start.split())).to(device)
    # tgt_mask = model.generate_square_subsequent_mask(len(start.split())).to(device)
    # src_mask = model.generate_square_subsequent_mask(src.size(0)).to(device)
    # tgt_mask = model.generate_square_subsequent_mask(tgt.size(0)).to(device)

    src_mask = model.generate_square_subsequent_mask(start_tens.size(0)).to(device)
    tgt_mask = model.generate_square_subsequent_mask(tgt_tens.size(0)).to(device)

    # src_mask = model.generate_square_subsequent_mask(len(start.split())).to(device)

    # mask = (torch.ones(num_words, num_words) == 1).transpose(0, 1)
    # mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)).to(device)
    # print("src_mask size: {}".format(src_mask.size()))
    # print(start_tens.size())

    # TODO make sure this transfers from above well
    out = model(start_tens, src_mask, tgt_tens, tgt_mask)

    # out = model(start_tens, mask)
    out = out.view(-1, ntokens)
    # print(out.size())
    # print(out.view(-1, ntokens).size())
    # TODO so I'm not really sure this output can be considered as log probabilities, its just an encoder? but maybe
    out = F.softmax(out, 1)
    # out_ind = torch.max(out, 1)[1].long()
    out_ind = torch.multinomial(torch.exp(out / temperature), 1).long()
    # print(out_ind)
    # print("And finally:")
    out_strings = decode_tensor_to_string(out_ind[:len(start.split())], vocab)
    new_string = out_strings.split()[-1]
    # print(out_strings)
    # print(start +" "+ new_string)
    return new_string
    # print(decode_tensor_to_string(out_ind[:len(start.split())], vocab))

# start = "choose a creature within range"
start = "choose a creature"
# start = "make a melee spell attack"
# start = "Quoth the raven, "
temp = 0.15
for i in range(10):
    start += " " + pick_next_word(start, vocab, model, temperature=temp)

final = start
print(final)