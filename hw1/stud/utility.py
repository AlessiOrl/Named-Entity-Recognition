##IMPORTS##
from typing import List, Optional
import numpy as np
from typing import List, Tuple

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
from torch import BoolTensor, FloatTensor, LongTensor

##Other
import time
import random
import json
from typing import Dict
from torch.utils.data import Dataset, DataLoader
import itertools
import string
from gensim.models import KeyedVectors


def create_pos_vocab() -> Dict:
    """
    create pos vocabulary based on nltk universal postag
    :return: Dict of all possible postag with each index
    """

    pos_list = ["ADJ", "ADP", "PUNCT", "ADV", "AUX", "SYM", "INTJ", "CCONJ", "X", "NOUN", "DET", "PROPN", "NUM", "VERB",
                "PART", "PRON", "SCONJ", "<UNK>", "<PAD>", "<SEP>"]
    pos_index = dict()

    for seed, vocab in enumerate(pos_list):
        pos_index[vocab] = seed
    return pos_index


def create_char_vocab() -> Dict:
    """
    create char vocabulary based on string.printable
    :return: Dict of all printable chars with each index
    """
    char_index = dict()
    max_index = -1
    for i, printable_char in enumerate(string.printable):
        char_index[printable_char] = i
        max_index = max(max_index, i)
    char_index["<UNK>"] = max_index + 1
    char_index["<PAD>"] = max_index + 2
    char_index["<SEP>"] = max_index + 3

    return char_index


class StudentParams:
    """
    create Class for parameters implementation
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    additionals_datasets = ["WNUT17.tsv", "CoNLL.tsv"]
    additional_dataset = True
    lemming = False

    word_vocab_name = os.path.join("../..", "data", "embeddings", "glove.6B.300d.w2v.bin")
    word_vocab = KeyedVectors.load_word2vec_format(word_vocab_name, binary=True)
    word_vocab_size = len(word_vocab)
    word_embedding_dim = word_vocab.vector_size

    pos_vocab_index = create_pos_vocab()
    char_vocab_index = create_char_vocab()

    pos_vocab_size = len(pos_vocab_index)
    pos_embedding_dim = 20

    char_vocab_size = len(char_vocab_index)
    char_embedding_dim = 50

    word_vocab["<UNK>"] = np.random.random(word_embedding_dim)
    word_vocab["<PAD>"] = np.zeros(word_embedding_dim)

    num_classes = 14

    # GLOBAL EMBEDDING LAYER
    global_vector = True
    global_hidden_dim = 100
    global_lstm_layers = 2
    global_dropout = 0.30
    global_bidir = True

    # CHAR EMBEDDING LAYER
    char = True
    char_hidden_dim = 50
    char_lstm_layers = 3
    char_dropout = 0.30
    char_bidir = True

    # WORD EMBEDDING LAYER
    words_freeze = True

    # COMBINED LSTM LAYER
    combined_hidden_dim = 350
    combined_lstm_layers = 3
    combined_dropout = 0.35
    combined_bidir = True
    feature_dim_out = 350

    # CLASSIFICATOR LAYER
    crf = True
    classificator_dropout = 0.4

    # CoNLL PARAMETERS
    ## Parameters for traning with CoNLL dataset
    conll_num_classes = 10
    conll_classificator_dropout = 0.4

    # OTHER HYPERPARAMS
    epochs = 15
    batch_size = 64
    learning_rate = 0.0005  # default 0.001
    weight_decay = 1e-4

    global_embedding_dim = word_embedding_dim + char_hidden_dim * (1 if char else 0) * (
        2 if char_bidir else 1) + pos_embedding_dim
    full_embedding_dim = global_hidden_dim * (1 if global_vector else 0) * (
        2 if global_bidir else 1) + global_embedding_dim

    def __init__(self, device="cpu"):
        self.device = device


class StudDataset(Dataset):
    """
    Dataset Class for Homework1
    :param tokens: List of sentences where each sentence is a list of tokens
    :param vocab: vocabulary of word for indexing
    :param pos_vocab: vocabulary of pos for indexing
    :param char_vocab: vocabulary of chars for indexing
    :param labels: labels for the input tokens (optional)
    :param lemming: Boolean value for lemming
    :param conll_dataset: Boolean value to chage classes
    :return: Dict of all printable chars with each index
    """

    def __init__(self, tokens: List[List], vocab: Dict, pos_vocab: Dict, char_vocab: Dict,
                 labels: Optional[List] = None,
                 lemming: Optional[bool] = False,
                 lowercase: bool = False, conll_dataset=False, device: str = "cpu"):
        if labels is None:
            labels = []
        self.lowercase = lowercase
        self.lemming = lemming
        self.device = device
        self.encoded_data = []
        self.vocab = vocab
        self.pos_vocab = pos_vocab
        self.char_vocab = char_vocab
        self.conll_dataset = conll_dataset
        self.init_structures(tokens, labels)
        self.words_pad = self.vocab["<PAD>"]
        self.char_pad = self.char_vocab["<PAD>"]
        self.pos_pad = self.pos_vocab["<PAD>"]
        self.label_pad = self.encode_class("<PAD>", self.conll_dataset)

    def init_structures(self, s_tokens, s_labels) -> None:
        """
        Init class structures
        :param s_tokens: List of sentences where each sentence is a list of tokens
        :param s_labels: List of sentences where each sentence is a list of labels
        """

        for index_s, tokenized_s in enumerate(s_tokens):
            words_posses = nltk.pos_tag(tokenized_s, tagset='universal')

            words_idx, pos_idx, char_idx, words_len = self.index_words(words_posses)

            if len(s_labels) > 0:
                labels_idx = [StudDataset.encode_class(label, self.conll_dataset) for label in s_labels[index_s]]
            else:
                labels_idx = []

            words_idx = torch.tensor(words_idx)
            pos_idx = torch.tensor(pos_idx)
            labels_idx = torch.tensor(labels_idx)
            char_idx = torch.tensor(char_idx)
            words_len = torch.tensor(words_len)
            self.encoded_data.append(({'words_indx': words_idx, 'poses_indx': pos_idx,
                                       'chars_indx': (char_idx, words_len)}, labels_idx, index_s))

    def index_words(self, words: List[Tuple]) -> Tuple[list, list, list, list]:
        """
        Indexes of sentence, pos, chars, words
        :param words: List of tuple (token, pos)
        :returns: tuple of indexes in this order: sentences, poses, chars, words
        """

        idxs = []
        pos_idxs = []
        char_word_idxs = []
        words_indexes = []
        lemmatizer = WordNetLemmatizer()

        for i, (word, pos) in enumerate(words):

            if self.lemming: word = lemmatizer.lemmatize(word, pos=self.get_wordnet_pos(pos))

            if word in self.vocab.keys():
                idxs.append(self.vocab[word])
            else:
                idxs.append(self.vocab["<UNK>"])
            pos_idxs.append(self.pos_vocab.get(pos, self.pos_vocab.get('<UNK>')))

            for char in word:
                char_word_idxs.append(self.char_vocab.get(char, self.char_vocab.get("<UNK>")))
            words_indexes += [i] * len(word)

        return idxs, pos_idxs, char_word_idxs, words_indexes

    @staticmethod
    def decode_class(index, conll=False) -> str:
        """
        Function to decode class
        :param index: index of the class
        :return: The corresponding string class
        """
        label = StudDataset.get_class_labels(index_to_label=True, conll_dataset=conll)[index]
        return label if label != "<PAD>" else "O"

    @staticmethod
    def encode_class(label, conll=False) -> int:
        """
        Function to encode class
        :param label: the class in string format
        :return: The corresponding index value
        """
        return StudDataset.get_class_labels(conll_dataset=conll)[label]

    @staticmethod
    def get_class_labels(index_to_label: bool = False, conll_dataset: bool = False) -> Dict:
        """
        Function to get all the class label for this dataset
        :return: l
        """

        conll_d = {"B-ORG": 0, "B-LOC": 1, "B-MISC": 2, "B-PER": 3, "I-MISC": 4, "I-LOC": 5,
                   "I-ORG": 6, "I-PER": 7, "O": 8, "<PAD>": 9}
        inverted_conll_d = {0: "B-ORG", 1: "B-LOC", 2: "B-MISC", 3: "B-PER", 4: "I-MISC", 5: "I-LOC",
                            6: "I-ORG", 7: "I-PER", 8: "O", 9: "<PAD>"}

        d = {"B-CORP": 0, "B-CW": 1, "B-GRP": 2, "B-LOC": 3, "B-PER": 4, "B-PROD": 5, "I-CORP": 6, "I-CW": 7,
             "I-GRP": 8, "I-LOC": 9, "I-PER": 10, "I-PROD": 11, "O": 12, "<PAD>": 13}
        inverted_d = {0: "B-CORP", 1: "B-CW", 2: "B-GRP", 3: "B-LOC", 4: "B-PER", 5: "B-PROD", 6: "I-CORP", 7: "I-CW",
                      8: "I-GRP", 9: "I-LOC", 10: "I-PER", 11: "I-PROD", 12: "O", 13: "<PAD>"}
        default = inverted_d if index_to_label else d
        conll = inverted_conll_d if index_to_label else conll_d
        return conll if conll_dataset else default

    @staticmethod
    def get_wordnet_pos(treebank_tag):
        """
        Return wordnet tagset based on the given tag
        :return:
        """
        if treebank_tag.startswith('ADJ'):
            return wordnet.ADJ
        elif treebank_tag.startswith('VERB'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('ADV'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    def __len__(self) -> int:
        """
        Function that returns the number of samples of the dataset
        :return: returns len of encoded data as integer
        """
        return len(self.encoded_data)

    def getData(self) -> List:
        """
        Function to return the encoded dataset as List
        :return: List containing the encoded data
        """
        return self.encoded_data

    def __getitem__(self, idx: int) -> Tuple[Dict, List, int]:
        """
        Function to get a specific element of the dataset
        :param idx: index of the element
        :return: the element of the corresponding input
        """
        return self.encoded_data[idx]

    def collate_fn(self, data) -> Tuple[Dict, Dict]:
        """
        Function that is performed on each batch
        :return: a tuple formed by 2 element, one with the information of X and the other of Y
        """
        words, posses, chars, scattered_chars, ys = [], [], [], [], []
        # Unpack the encoded data
        for e in data:
            words.append(e[0]["words_indx"])
            posses.append(e[0]["poses_indx"])
            chars.append(e[0]["chars_indx"][0])
            scattered_chars.append(e[0]["chars_indx"][1])
            ys.append(e[1])

        # Pad each input list to the max_len of each batch
        words = torch.nn.utils.rnn.pad_sequence(words, padding_value=self.words_pad, batch_first=True)
        posses = torch.nn.utils.rnn.pad_sequence(posses, padding_value=self.pos_pad, batch_first=True)
        chars = torch.nn.utils.rnn.pad_sequence(chars, padding_value=self.char_pad, batch_first=True)
        scattered_chars = torch.nn.utils.rnn.pad_sequence(scattered_chars, padding_value=-1, batch_first=True)

        # Create a scatter matrix for the char list
        scattered_chars[scattered_chars == -1] = torch.max(scattered_chars) + 1

        ys = torch.nn.utils.rnn.pad_sequence(ys, padding_value=self.label_pad, batch_first=True)
        mask = words != self.vocab["<PAD>"]

        return {"words": words, "poses": posses, "chars": chars, "mask": mask, "scattered": scattered_chars}, {
            "labels": ys}


#################################################################
#                               					        	#
#       This code is taken from the repository Pytorch_CRF  	#
#        Code at: https://github.com/epwalsh/pytorch-crf        #
#                               					        	#
#################################################################

class CRF(nn.Module):

    def __init__(self, num_labels: int, pad_idx: Optional[int] = None, use_gpu: bool = False) -> None:
        """
		:param num_labels: number of labels
		:param pad_idxL padding index. default None
		:return None
		"""

        if num_labels < 1:
            raise ValueError("invalid number of labels: {0}".format(num_labels))

        super().__init__()
        self.num_labels = num_labels
        self._use_gpu = torch.cuda.is_available() and use_gpu

        # transition matrix setting
        # transition matrix format (source, destination)
        self.trans_matrix = nn.Parameter(torch.empty(num_labels, num_labels))
        # transition matrix of start and end settings
        self.start_trans = nn.Parameter(torch.empty(num_labels))
        self.end_trans = nn.Parameter(torch.empty(num_labels))

        self._initialize_parameters(pad_idx)

    def forward(self, h: FloatTensor, labels: LongTensor, mask: BoolTensor) -> FloatTensor:
        """
		:param h: hidden matrix (batch_size, seq_len, num_labels)
		:param labels: answer labels of each sequence
									 in mini batch (batch_size, seq_len)
		:param mask: mask tensor of each sequence
								 in mini batch (batch_size, seq_len)
		:return: The log-likelihood (batch_size)
		"""

        log_numerator = self._compute_numerator_log_likelihood(h, labels, mask)
        log_denominator = self._compute_denominator_log_likelihood(h, mask)

        return log_numerator - log_denominator

    def viterbi_decode(self, h: FloatTensor, mask: BoolTensor) -> List[List[int]]:
        """
		decode labels using viterbi algorithm
		:param h: hidden matrix (batch_size, seq_len, num_labels)
		:param mask: mask tensor of each sequence
								 in mini batch (batch_size, batch_size)
		:return: labels of each sequence in mini batch
		"""

        batch_size, seq_len, _ = h.size()
        # prepare the sequence lengths in each sequence
        seq_lens = mask.sum(dim=1)
        # In mini batch, prepare the score
        # from the start sequence to the first label
        score = [self.start_trans.data + h[:, 0]]
        path = []

        for t in range(1, seq_len):
            # extract the score of previous sequence
            # (batch_size, num_labels, 1)
            previous_score = score[t - 1].view(batch_size, -1, 1)

            # extract the score of hidden matrix of sequence
            # (batch_size, 1, num_labels)
            h_t = h[:, t].view(batch_size, 1, -1)

            # extract the score in transition
            # from label of t-1 sequence to label of sequence of t
            # self.trans_matrix has the score of the transition
            # from sequence A to sequence B
            # (batch_size, num_labels, num_labels)
            score_t = previous_score + self.trans_matrix + h_t

            # keep the maximum value
            # and point where maximum value of each sequence
            # (batch_size, num_labels)
            best_score, best_path = score_t.max(1)
            score.append(best_score)
            path.append(best_path)

        # predict labels of mini batch
        best_paths = [self._viterbi_compute_best_path(i, seq_lens, score, path) for i in range(batch_size)]

        return best_paths

    def _viterbi_compute_best_path(self, batch_idx: int, seq_lens: torch.LongTensor, score: List[FloatTensor],
                                   path: List[torch.LongTensor], ) -> List[int]:
        """
		return labels using viterbi algorithm
		:param batch_idx: index of batch
		:param seq_lens: sequence lengths in mini batch (batch_size)
		:param score: transition scores of length max sequence size
									in mini batch [(batch_size, num_labels)]
		:param path: transition paths of length max sequence size
								 in mini batch [(batch_size, num_labels)]
		:return: labels of batch_idx-th sequence
		"""

        seq_end_idx = seq_lens[batch_idx] - 1
        # extract label of end sequence
        _, best_last_label = (score[seq_end_idx][batch_idx] + self.end_trans).max(0)
        best_labels = [int(best_last_label)]

        # predict labels from back using viterbi algorithm
        for p in reversed(path[:seq_end_idx]):
            best_last_label = p[batch_idx][best_labels[0]]
            best_labels.insert(0, int(best_last_label))

        return best_labels

    def _compute_denominator_log_likelihood(self, h: FloatTensor, mask: BoolTensor):
        """
		compute the denominator term for the log-likelihood
		:param h: hidden matrix (batch_size, seq_len, num_labels)
		:param mask: mask tensor of each sequence
								 in mini batch (batch_size, seq_len)
		:return: The score of denominator term for the log-likelihood
		"""
        device = h.device
        batch_size, seq_len, _ = h.size()

        # (num_labels, num_labels) -> (1, num_labels, num_labels)
        trans = self.trans_matrix.unsqueeze(0)

        # add the score from beginning to each label
        # and the first score of each label
        score = self.start_trans + h[:, 0]

        # iterate through processing for the number of words in the mini batch
        for t in range(1, seq_len):
            # (batch_size, self.num_labels, 1)
            before_score = score.unsqueeze(2)

            # prepare t-th mask of sequences in each sequence
            # (batch_size, 1)
            mask_t = mask[:, t].unsqueeze(1)
            mask_t = mask_t.to(device)

            # prepare the transition probability of the t-th sequence label
            # in each sequence
            # (batch_size, 1, num_labels)
            h_t = h[:, t].unsqueeze(1)

            # calculate t-th scores in each sequence
            # (batch_size, num_labels)
            score_t = before_score + h_t + trans
            score_t = torch.logsumexp(score_t, 1)

            # update scores
            # (batch_size, num_labels)
            score = torch.where(mask_t, score_t, score)

        # add the end score of each label
        score += self.end_trans

        # return the log likely food of all data in mini batch
        return torch.logsumexp(score, 1)

    def _compute_numerator_log_likelihood(self, h: FloatTensor, y: LongTensor, mask: BoolTensor) -> FloatTensor:
        """
		compute the numerator term for the log-likelihood
		:param h: hidden matrix (batch_size, seq_len, num_labels)
		:param y: answer labels of each sequence
							in mini batch (batch_size, seq_len)
		:param mask: mask tensor of each sequence
								 in mini batch (batch_size, seq_len)
		:return: The score of numerator term for the log-likelihood
		"""

        batch_size, seq_len, _ = h.size()

        h_unsqueezed = h.unsqueeze(-1)
        trans = self.trans_matrix.unsqueeze(-1)

        arange_b = torch.arange(batch_size)

        # extract first vector of sequences in mini batch
        calc_range = seq_len - 1
        score = self.start_trans[y[:, 0]] + sum(
            [self._calc_trans_score_for_num_llh(h_unsqueezed, y, trans, mask, t, arange_b) for t in range(calc_range)])

        # extract end label number of each sequence in mini batch
        # (batch_size)
        last_mask_index = mask.sum(1) - 1
        last_labels = y[arange_b, last_mask_index]
        each_last_score = h[arange_b, -1, last_labels] * mask[:, -1]

        # Add the score of the sequences of the maximum length in mini batch
        # Add the scores from the last tag of each sequence to EOS
        score += each_last_score + self.end_trans[last_labels]

        return score

    def _calc_trans_score_for_num_llh(self, h: FloatTensor, y: LongTensor, trans: FloatTensor, mask: BoolTensor, t: int,
                                      arange_b: FloatTensor, ) -> torch.Tensor:
        """
		calculate transition score for computing numberator llh
		:param h: hidden matrix (batch_size, seq_len, num_labels)
		:param y: answer labels of each sequence
							in mini batch (batch_size, seq_len)
		:param trans: transition score
		:param mask: mask tensor of each sequence
								 in mini batch (batch_size, seq_len)
		:paramt t: index of hidden, transition, and mask matrixex
		:param arange_b: this param is seted torch.arange(batch_size)
		:param batch_size: batch size of this calculation
		"""
        device = h.device
        mask_t = mask[:, t]
        mask_t = mask_t.to(device)
        mask_t1 = mask[:, t + 1]
        mask_t1 = mask_t1.to(device)

        # extract the score of t+1 label
        # (batch_size)
        h_t = h[arange_b, t, y[:, t]].squeeze(1)

        # extract the transition score from t-th label to t+1 label
        # (batch_size)
        trans_t = trans[y[:, t], y[:, t + 1]].squeeze(1)

        # add the score of t+1 and the transition score
        # (batch_size)
        return h_t * mask_t + trans_t * mask_t1

    def _initialize_parameters(self, pad_idx: Optional[int]) -> None:
        """
		initialize transition parameters
		:param: pad_idx: if not None, additional initialize
		:return: None
		"""

        nn.init.uniform_(self.trans_matrix, -0.1, 0.1)
        nn.init.uniform_(self.start_trans, -0.1, 0.1)
        nn.init.uniform_(self.end_trans, -0.1, 0.1)
        if pad_idx is not None:
            self.start_trans[pad_idx] = -10000.0
            self.trans_matrix[pad_idx, :] = -10000.0
            self.trans_matrix[:, pad_idx] = -10000.0
            self.trans_matrix[pad_idx, pad_idx] = 0.0


#################################################################
#                               					        	#
#    This code is taken from the repository pytorch-scatter  	#
#        Code at: https://github.com/epwalsh/pytorch-crf        #
#                               					        	#
#################################################################

def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand(other.size())
    return src


def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None) -> torch.Tensor:
    index = broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        return out.scatter_add_(dim, index, src)
    else:
        return out.scatter_add_(dim, index, src)


def scatter_add(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None) -> torch.Tensor:
    return scatter_sum(src, index, dim, out, dim_size)


def scatter_mul(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None) -> torch.Tensor:
    return torch.ops.torch_scatter.scatter_mul(src, index, dim, out, dim_size)


def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                 out: Optional[torch.Tensor] = None,
                 dim_size: Optional[int] = None) -> torch.Tensor:
    out = scatter_sum(src, index, dim, out, dim_size)
    dim_size = out.size(dim)

    index_dim = dim
    if index_dim < 0:
        index_dim = index_dim + src.dim()
    if index.dim() <= index_dim:
        index_dim = index.dim() - 1

    ones = torch.ones(index.size(), dtype=src.dtype, device=src.device)
    count = scatter_sum(ones, index, index_dim, None, dim_size)
    count[count < 1] = 1
    count = broadcast(count, out, dim)
    if out.is_floating_point():
        out.true_divide_(count)
    else:
        out.div_(count, rounding_mode='floor')
    return out
