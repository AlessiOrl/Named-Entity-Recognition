import imp
import numpy as np
from typing import List, Tuple

from model import Model

## for data
import os
import numpy as np

## for plotting
import matplotlib.pyplot as plt

## for processing
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

## for word embedding
import gensim
import gensim.downloader as gensim_api

## for model
import torch
import torch.nn as nn

##Other
import time
import random
import json
from typing import Dict
from torch.utils.data import Dataset, DataLoader
import itertools
import string

from .utility import StudDataset
from .utility import CRF
from .utility import scatter_sum
from .utility import StudentParams

nltk.download('wordnet')
nltk.download('universal_tagset')
nltk.download('averaged_perceptron_tagger')

def build_model(device: str) -> Model:
	# STUDENT: return StudentModel()
	# STUDENT: your model MUST be loaded on the device "device" indicates
	model_p = os.path.join("model", "model.ckpt")
	params = StudentParams(device = device)
	model = StudentModel(params)
	model.load_state_dict(torch.load(model_p, map_location=device))
	model.to(device)

	return model



class RandomBaseline(Model):
	options = [(3111, "B-CORP"), (3752, "B-CW"), (3571, "B-GRP"), (4799, "B-LOC"), (5397, "B-PER"), (2923, "B-PROD"), (3111, "I-CORP"), (6030, "I-CW"), (6467, "I-GRP"), (2751, "I-LOC"), (6141, "I-PER"), (1800, "I-PROD"), (203394, "O")]

	def __init__(self):
		self._options = [option[1] for option in self.options]
		self._weights = np.array([option[0] for option in self.options])
		self._weights = self._weights / self._weights.sum()

	def predict(self, tokens: List[List[str]]) -> List[List[str]]:
		return [[str(np.random.choice(self._options, 1, p=self._weights)[0]) for _x in x] for x in tokens]


class StudentModel(Model, nn.Module):
	def __init__(self, params: StudentParams):
		super(StudentModel, self).__init__()
		self.params = params
		self.device = params.device
		
		# EMBEDDING LAYERS
		self.word_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(self.params.word_vocab.vectors), freeze=self.params.words_freeze, padding_idx=self.params.word_vocab.key_to_index["<PAD>"])
		self.pos_embedding = nn.Embedding(num_embeddings=self.params.pos_vocab_size, embedding_dim=self.params.pos_embedding_dim, padding_idx=self.params.pos_vocab_index["<PAD>"])
		self.char_embedding = nn.Embedding(num_embeddings=self.params.char_vocab_size, embedding_dim=self.params.char_embedding_dim, padding_idx=self.params.char_vocab_index["<PAD>"])

		# LAYER 1.1: Char BiLSTM
		self.char_lstm = nn.LSTM(input_size=self.params.char_embedding_dim, hidden_size=self.params.char_hidden_dim, num_layers=self.params.char_lstm_layers, bidirectional=self.params.char_bidir, dropout=self.params.char_dropout if self.params.char_lstm_layers > 1 else 0, batch_first=True)
		self.dense_char_embedding = nn.Linear(self.params.char_hidden_dim * (2 if self.params.char_bidir else 1), self.params.char_hidden_dim * (2 if self.params.char_bidir else 1))
		self.char_dropout = nn.Dropout(self.params.char_dropout)

		# LAYER 1.2: Global BiLSTM
		self.global_contextual_embedding = nn.LSTM(input_size=self.params.global_embedding_dim, hidden_size=self.params.global_hidden_dim, num_layers=self.params.global_lstm_layers, bidirectional=self.params.global_bidir, dropout=self.params.global_dropout if self.params.global_lstm_layers > 1 else 0, batch_first=True)

		self.dense_global_embedding = nn.Linear(self.params.global_hidden_dim * (2 if self.params.global_bidir else 1), self.params.global_hidden_dim * (2 if self.params.global_bidir else 1))
		self.global_dropout = nn.Dropout(self.params.global_dropout)

		# LAYER 2: Feature BiLSTM
		self.combined_lstm = nn.LSTM(input_size=self.params.full_embedding_dim, hidden_size=self.params.combined_hidden_dim, num_layers=self.params.combined_lstm_layers, bidirectional=self.params.combined_bidir, dropout=self.params.combined_dropout if self.params.combined_lstm_layers > 1 else 0, batch_first=True)

		self.dense_combined_embedding = nn.Linear(self.params.combined_hidden_dim * (2 if self.params.combined_bidir else 1), self.params.feature_dim_out)
		self.combined_dropout = nn.Dropout(self.params.combined_dropout)

		# LAYER 3.1: Classificator
		self.batchnorm = nn.BatchNorm1d(self.params.feature_dim_out)
		self.SELU = nn.SELU()

		self.fc = nn.Linear(self.params.feature_dim_out, 128)
		self.fc2 = nn.Linear(128, self.params.num_classes)
		self.fc_dropout = nn.Dropout(self.params.classificator_dropout)

		# LAYER 3.2: Classificator
		self.conll_fc = nn.Linear(self.params.feature_dim_out, self.params.conll_num_classes)
		self.conll_fc_dropout = nn.Dropout(self.params.conll_classificator_dropout)

		self.softmax = nn.LogSoftmax(dim=1)

		# LAYER 4
		if self.params.crf:
			self.crf = CRF(self.params.num_classes).to(self.device)
			# LAYER 4.2
			self.conll_crf = CRF(self.params.conll_num_classes).to(self.device)

	def ner_loss(self, y_pred, y, mask, criterion=None, conll=False):
		if criterion is None:
			criterion = nn.CrossEntropyLoss(ignore_index=StudDataset.encode_class("<PAD>"), reduction="mean")

		if self.params.crf:
			if conll:
				loss = -(self.conll_crf(y_pred, y, mask)).mean()
			else:
				loss = -(self.crf(y_pred, y, mask)).mean()
		else:
			# labels  [[1,2,3], [18, 12, 3]] after the view(-1) [1,2,3, 18, 12, 3]
			y_pred = y_pred.view(-1, y_pred.shape[-1])
			y = y.view(-1)
			# FLATTENED MASK
			f_mask = mask.view(-1)

			# FILTER NOT PADDING
			y_pred = y_pred[f_mask]
			y = y[f_mask]
			loss = criterion(y_pred, y)
		return loss

	def forward(self, x, conll=False, verbose=False):

		# Unpack chars and sentence_char_len
		words = x[0]
		poses = x[1]
		chars = x[2]
		char_indexes = x[3]
		mask = x[4]

		# Embedding layers
		poses_out = self.pos_embedding(poses)
		words_out = self.word_embedding(words)

		# FIRST LSTM layer
		if self.params.char:
			chars_out = self.char_embedding(chars)
			char_lstm_out, _ = self.char_lstm(chars_out)

			# char_lstm_out = self.dense_char_embedding(char_lstm_out)
			char_lstm_out = self.char_dropout(char_lstm_out)

			if verbose: print("FIRST lstm out shape: {}".format(char_lstm_out.shape))
			# COMBINE CHARS FOR EACH WORDS

			batch_sentences = scatter_sum(char_lstm_out, char_indexes, dim=1)[:, :words_out.shape[1], :]

			if verbose: print("Scatter out shape: {}".format(batch_sentences.shape))
			if verbose: print("words out shape: {}".format(words_out.shape))
			if verbose: print("poses out shape: {}".format(poses_out.shape))

			combined_embeddings = torch.cat((words_out, batch_sentences, poses_out), 2)
		else:
			combined_embeddings = torch.cat((words_out, poses_out), 2)

		if self.params.global_vector:
			# Global Contextual Embedding

			global_lstm_out, _ = self.global_contextual_embedding(combined_embeddings)
			global_lstm_out = self.dense_global_embedding(global_lstm_out)
			global_lstm_out = self.global_dropout(global_lstm_out)

			global_context_emb = torch.sum(global_lstm_out, dim=1)
			global_context_emb = torch.unsqueeze(global_context_emb, dim=1)

			global_context_emb = global_context_emb.expand(-1, global_lstm_out.shape[1], -1)

			if verbose: print("global_context_emb embeddings shape: {}".format(global_context_emb.shape))

		if self.params.global_vector:
			combined_embeddings = torch.cat((global_context_emb, combined_embeddings), 2)

		if verbose: print("COMBINED embeddings shape: {}".format(combined_embeddings.shape))

		# SECOND LSTM LAYER
		full_lstm_out, _ = self.combined_lstm(combined_embeddings)

		if verbose: print("SECOND lstm shape: {}".format(full_lstm_out.shape))
		feature_lstm_out = self.dense_combined_embedding(full_lstm_out)
		feature_lstm_out = self.combined_dropout(feature_lstm_out)

		if verbose: print("FEATURE extractor shape: {}".format(feature_lstm_out.shape))
		feature_lstm_out = self.batchnorm(feature_lstm_out.permute(0, 2, 1))

		if verbose: print("BATCHNORM shape: {}".format(feature_lstm_out.shape))

		activation_function = self.SELU

		out = activation_function(feature_lstm_out.permute(0, 2, 1))

		# IF IS USED THE CONLL DATASET USE A CLASSIFICATOR WITH LESS CLASSES
		if conll:
			out = self.conll_fc(out)

		else:
			out = self.fc(out)
			out = self.fc_dropout(out)
			out = self.fc2(out)

		logits = self.softmax(out)
		if verbose: print("LOGITS shape: {}".format(out.shape))

		if self.params.crf:
			if conll:
				out = self.conll_crf.viterbi_decode(logits, mask)
			else:
				out = self.crf.viterbi_decode(logits, mask)
		return logits, out

	def predict(self, tokens: List[List[str]], conll = False) -> List[List[str]]:
		all_predict = []
		dev_dataset = StudDataset(tokens, self.params.word_vocab.key_to_index, self.params.pos_vocab_index, self.params.char_vocab_index, device=self.params.device, conll_dataset=conll)

		dataloader = DataLoader(dev_dataset, batch_size=self.params.batch_size, collate_fn=dev_dataset.collate_fn)
		self.eval()

		with torch.no_grad():
			for (xd, ys) in dataloader:
				mask = xd["mask"].to(self.device)
				x = (xd["words"].to(self.device), xd["poses"].to(self.device), xd["chars"].to(self.device), xd["scattered"].to(self.device), mask)

				hidden, out = self(x,conll)

				### START EVAL PART ###
				for s_ine, sentence in enumerate(out):
					all_predict.append([StudDataset.decode_class(x, conll) for x in sentence])
				### END EVAL PART ###

		return all_predict
