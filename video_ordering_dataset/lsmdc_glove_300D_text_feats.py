import numpy as np
import nltk

import pdb
import pandas as pd
from tqdm import tqdm

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# python3.5 lsmdc_glove_300D_text_feats.py


# This function creates a normalized vector for the whole sentence
def sent2vec(s):
	words = str(s).lower()
	words = word_tokenize(words)
	words = [w for w in words if not w in stop_words]
	words = [w for w in words if w.isalpha()]
	M = []
	for w in words:
		try:
			M.append(embeddings_index[w])
		except:
			continue
	M = np.array(M)
	v = M.sum(axis=0)
	if type(v) != np.ndarray:
		return np.zeros(300)
	return v / np.sqrt((v ** 2).sum())

# load the GloVe vectors in a dictionary:
# Model used: glove.6B.300d
path_to_glove = '/home/vsharma/Documents/triplet-network-pytorch/models/glove.6B.300d.txt'
embeddings_index = {}
f = open(path_to_glove)
for line in tqdm(f):
	values = line.split()
	word = values[0]
	coefs = np.asarray(values[1:], dtype='float32')
	embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))


# Loading LSMDC Subtiles
LSMDC_Subtitles = '/cvhci/data/QA/Movie_Description_Dataset/subtitles/LSMDC_Subtitles-oneFile.txt'
df = pd.read_csv('{}'.format(LSMDC_Subtitles), sep='\t', header=None)

# Index 1 contains the subtitles
subtitles = df[1]
num_samples= subtitles.shape

# create sentence vectors using the above function for training and validation set
feat_glove_lsmdc = [sent2vec(line_) for line_ in tqdm(subtitles)]

np.save('feats_glove_lsmdc.npy',feat_glove_lsmdc)