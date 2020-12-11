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
        print("d_model: {}".format(d_model))
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