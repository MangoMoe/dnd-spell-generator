# Notes

Be sure to specify that I have legal access to the training data (I own the books online)

## Plans
* I think I will start by training a sequence-to-sequence model. Then I will generate new spells by putting in random data for the vector that goes in between.
    * should I use a transformer for this or focus on the NMT architecture or what?
    * I think I'll go with GRU/LSTM then try Transformer, then move from there.
* TODO check out the regularization slides
1. GRU architecture
    * TODO stop trying to get the broken lab to work, use a tutorial or do it from scratch
    * How to handle the start and stop tokens
        * **LOOK AT THE TRANSFORMER LAB in the data loading section**
            * Also look at the torchtext functions in the same place
        * Well if we are using word embeddings, then we can just use something like all zeros and all ones for a start and stop vector, the likelihood of collision is low.
        * How do you use embeddings again???
            * nn.Embedding
2. Transformers
    * Don't forget about positional encoding
3. BERT or some other method