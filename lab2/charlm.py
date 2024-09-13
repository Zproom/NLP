from collections import *
from math import *
import pprint
import operator
from random import random

# charlm.py: exmaple code for lab 2 in 605.646

# This lab is inspired from a similar lab of Chris Callison-Burch's,
# which was in turn inspired by a blot post by Yoav Goldberg:
# https://nbviewer.jupyter.org/gist/yoavg/d76121dfde2618422139
#
# We fixed the example code to pad each sentence/line with a start indicator instead
# of only padding the very first sentence.

# Convert counts to probabilities for successor chars in a given context
def normalize(counter):
        s = float(sum(counter.values()))
        return [(c,cnt/s) for c,cnt in counter.items()]

# Read a training file and produce a language model
def train_char_lm(fname, order=4):
    data = open(fname).read()
    sents = data.split('\n')
    lm = defaultdict(Counter)
    for s in sents:
        pad = "~" * order
        data = pad + s + '\n'
        for i in range(len(data)-order):
            history, char = data[i:i+order], data[i+order]
            lm[history][char]+=1
    outlm = {hist:normalize(chars) for hist, chars in lm.items()}
    return outlm

# Given a character LM, randomly choose a next character given this history and return it
def generate_letter(lm, history, order):
        history = history[-order:]
        dist = lm[history]
        x = random()
        for c,v in dist:
            x = x - v
            if x <= 0: return c

# Generate a random text by repeatedly calling generate_letter
def generate_text(lm, order, nletters=1000):
    history = "~" * order
    out = []
    for i in range(nletters):
        c = generate_letter(lm, history, order)
        history = history[-order:] + c
        out.append(c)
        if c == '\n':
            history = "~" * order
    return "".join(out)

# Print alternatives given this context
def print_probs(lm, history):
    probs = sorted(lm[history],key=lambda x:(-x[1],x[0]))
    pp = pprint.PrettyPrinter()
    pp.pprint(probs)

# Compute the per-char perplexity of a text, using an input LM.  Returns infinity if a probability isn't found in the model
def perplexity(text, lm, order=4):
    # Pad the input with "~" chars.  This handles the case where order > len(text).
    pad = "~" * order
    data = pad + text
    # This is a stub.
    # Loop over data string and find probs and use to compute perplexity
    return float("inf")

# Computes per-char perplexity of a text, given an input LM.  Smoothing is very, very simple, just using a small constant
def smoothed_perplexity(text, lm, order=4):
    # Pad the input with "~" chars.  This handles the case where order > len(text).
    pad = "~" * order
    data = pad + text
    # This is a stub.
    # Loop over data string and find probs and use to compute perplexity
    return 0.001

# end of file
