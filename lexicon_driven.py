import re
import collections
from param import letter_string

def train(dic):
    model = collections.defaultdict(lambda: 1) #smoothing
    for f in dic:
        model[f] += 1
    return model

def edits1(word):
   splits     = [(word[:i], word[i:]) for i in range(len(word) + 1)]
   deletes    = [a + b[1:] for a, b in splits if b]
   transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b)>1]
   replaces   = [a + c + b[1:] for a, b in splits for c in letter_string if b]
   inserts    = [a + c + b     for a, b in splits for c in letter_string]
   return set(deletes + transposes + replaces + inserts)

def known_edits2(word,model):
    return set(e2 for e1 in edits1(word) for e2 in edits1(e1) if e2 in model)

def known(words,model): 
    return set(w for w in words if w in model)

def correct(word,model):
    candidates = known([word],model) or known(edits1(word),model) or known_edits2(word,model) or [word]
    return max(candidates, key=model.get)
