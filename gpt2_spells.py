# %%
import gpt_2_simple as gpt2
import glob

# %%
sess = gpt2.start_tf_sess()
# TODO if this doesn't work then congomerate all the spells into one file
# TODO change model size as well
for fil in glob.glob("spells/*.txt"):
    gpt2.finetune(sess,
        fil,
        model_name="124M",
        steps=1000)

# %%
print(gpt2.generate(sess))

# %%
# To load automatically saved checkpoint
sess = gpt2.start_tf_sess()
gpt2.load_gpt2(sess)