import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def random_embedding(vocab_size, embedding_dim):
    pretrain_emb = np.empty([vocab_size, embedding_dim])
    scale = np.sqrt(3.0 / embedding_dim)
    for index in range(vocab_size):
        pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedding_dim])
    return pretrain_emb


class CharBiLSTM(nn.Module):
    def __init__(self,
                 char_vocab_size,
                 pretrain_char_embedding,
                 char_emb_dim,
                 char_hidden_dim,
                 dropout,
                 char_feature_extractor):
        super(CharBiLSTM, self).__init__()
        char_hidden_dim = char_hidden_dim // 2
        self.dropout = nn.Dropout(dropout)
        self.char_embeddings = nn.Embedding(char_vocab_size, char_emb_dim)
        if pretrain_char_embedding is not None:
            self.char_embeddings.weight.data.copy_(torch.from_numpy(pretrain_char_embedding))
        else:
            self.char_embeddings.weight.data.copy_(torch.from_numpy(random_embedding(char_vocab_size, char_emb_dim)))
        if char_feature_extractor == "GRU":
            self.char_lstm = nn.GRU(char_emb_dim, char_hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        elif char_feature_extractor == "LSTM":
            self.char_lstm = nn.LSTM(char_emb_dim, char_hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        else:
            raise ValueError(f'unknown extractor {char_feature_extractor}!')

    def forward(self, input, seq_lengths):
        batch_size = input.size(0)
        char_embeds = self.dropout(self.char_embeddings(input))
        pack_input = pack_padded_sequence(char_embeds, seq_lengths, batch_first=True, enforce_sorted=False)
        char_out, char_hidden = self.char_lstm(pack_input, None)
        return char_hidden[0].transpose(1, 0).contiguous().view(batch_size, -1)


class CharCNN(nn.Module):
    def __init__(self,
                 char_vocab_size,
                 pretrain_char_embedding,
                 char_emb_dim,
                 char_hidden_dim,
                 dropout):
        super(CharCNN, self).__init__()
        self.dropout = dropout
        self.char_drop = nn.Dropout(dropout)
        self.char_embeddings = nn.Embedding(char_vocab_size, char_emb_dim)
        if pretrain_char_embedding is not None:
            self.char_embeddings.weight.data.copy_(torch.from_numpy(pretrain_char_embedding))
        else:
            self.char_embeddings.weight.data.copy_(torch.from_numpy(random_embedding(char_vocab_size, char_emb_dim)))
        self.char_cnn = nn.Conv1d(char_emb_dim, char_hidden_dim, kernel_size=3, padding=1)

    def forward(self, input, seq_lengths):
        """
            input:
                input: Variable(batch_size, word_length)
                seq_lengths: numpy array (batch_size,  1)
            output:
                Variable(batch_size, char_hidden_dim)
            Note it only accepts ordered (length) variable, length size is recorded in seq_lengths
        """
        batch_size = input.size(0)
        char_embeds = self.char_drop(self.char_embeddings(input))
        char_embeds = char_embeds.transpose(2, 1)
        char_cnn_out = self.char_cnn(char_embeds)
        char_cnn_out = F.max_pool1d(char_cnn_out, char_cnn_out.size(2)).view(batch_size, -1)
        return char_cnn_out


class WordRep(nn.Module):
    def __init__(self,
                 word_vocab_size,
                 char_vocab_size,
                 word_embedding,
                 word_emb_dim,
                 use_char,
                 char_embedding,
                 char_emb_dim,
                 char_hidden_dim,
                 char_feature_extractor,
                 dropout,
                 ):
        super(WordRep, self).__init__()
        self.use_char = use_char
        if use_char:
            self.char_all_feature = False
            if char_feature_extractor == "CNN":
                self.char_feature = CharCNN(char_vocab_size, char_embedding, char_emb_dim, char_hidden_dim, dropout)
            elif char_feature_extractor in ["LSTM", "GRU"]:
                self.char_feature = CharBiLSTM(char_vocab_size, char_embedding, char_emb_dim, char_hidden_dim, dropout, char_feature_extractor)
            elif char_feature_extractor in ["CNN-LSTM", "CNN-GRU"]:
                self.char_all_feature = True
                self.char_feature = CharCNN(char_vocab_size, char_embedding, char_emb_dim, char_hidden_dim, dropout)
                char_feature_extractor_extra = char_feature_extractor.split('-')[1]
                self.char_feature_extra = CharBiLSTM(char_vocab_size, char_embedding, char_emb_dim, char_hidden_dim, dropout, char_feature_extractor_extra)
            else:
                raise ValueError(f'unknown extractor {char_feature_extractor}!')
        self.dropout = nn.Dropout(dropout)
        self.word_embedding = nn.Embedding(word_vocab_size, word_emb_dim)
        if word_embedding is not None:
            assert word_embedding.shape[0] == word_vocab_size
            self.word_embedding.weight.data.copy_(torch.from_numpy(word_embedding))
        else:
            self.word_embedding.weight.data.copy_(torch.from_numpy(random_embedding(word_vocab_size, word_emb_dim)))

    def forward(self, word_inputs, char_inputs=None, char_seq_lengths=None):
        """
            input:
                word_inputs: (batch_size, sent_len)
                features: list [(batch_size, sent_len), (batch_len, sent_len),...]
                word_seq_lengths: list of batch_size, (batch_size,1)
                char_inputs: (batch_size*sent_len, word_length)
                char_seq_lengths: list of whole batch_size for char, (batch_size*sent_len, 1)
            output:
                Variable(batch_size, sent_len, hidden_dim)
        """
        batch_size = word_inputs.size(0)
        sent_len = word_inputs.size(1)
        word_embs = self.word_embedding(word_inputs)

        word_list = [word_embs]
        if self.use_char:
            assert char_inputs is not None and char_seq_lengths is not None
            ## calculate char lstm last hidden
            char_features = self.char_feature(char_inputs, char_seq_lengths)
            char_features = char_features.view(batch_size, sent_len, -1)
            ## concat word and char together
            word_list.append(char_features)
            if self.char_all_feature:
                char_features_extra = self.char_feature_extra(char_inputs, char_seq_lengths)
                # char_features_extra = char_features_extra[char_seq_recover]
                char_features_extra = char_features_extra.view(batch_size, sent_len, -1)
                ## concat word and char together
                word_list.append(char_features_extra)
        word_embs = torch.cat(word_list, 2)
        word_represent = self.dropout(word_embs)
        return word_represent


class WordSequence(nn.Module):
    def __init__(self,
                 word_vocab_size,
                 char_vocab_size,
                 dropout,
                 word_embedding,
                 word_emb_dim,
                 word_hidden_dim,
                 word_feature_extractor,
                 n_word_hidden_layer,
                 use_char,
                 char_embedding,
                 char_emb_dim,
                 char_hidden_dim,
                 char_feature_extractor):
        super(WordSequence, self).__init__()
        self.use_char = True
        input_size = word_emb_dim
        self.dropout = nn.Dropout(dropout)
        if use_char:
            self.wordrep = WordRep(
                word_vocab_size=word_vocab_size,
                char_vocab_size=char_vocab_size,
                word_embedding=word_embedding,
                word_emb_dim=word_emb_dim,
                use_char=use_char,
                char_embedding=char_embedding,
                char_emb_dim=char_emb_dim,
                char_hidden_dim=char_hidden_dim,
                char_feature_extractor=char_feature_extractor,
                dropout=dropout)
            input_size += char_hidden_dim
            if char_feature_extractor in ["CNN-LSTM", "CNN-GRU"]:
                input_size += char_hidden_dim
        else:
            self.wordrep = WordRep(
                word_vocab_size=word_vocab_size,
                char_vocab_size=char_vocab_size,
                word_embedding=word_embedding,
                word_emb_dim=word_emb_dim,
                use_char=False,
                char_embedding=None,
                char_emb_dim=0,
                char_hidden_dim=0,
                char_feature_extractor=None,
                dropout=dropout)
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        lstm_hidden = word_hidden_dim // 2  # args.HP_hidden_dim // 2

        if word_feature_extractor == "GRU":
            self.lstm = nn.GRU(input_size, lstm_hidden, num_layers=n_word_hidden_layer, batch_first=True, bidirectional=True)
        elif word_feature_extractor == "LSTM":
            self.lstm = nn.LSTM(input_size, lstm_hidden, num_layers=n_word_hidden_layer, batch_first=True, bidirectional=True)
        else:
            raise ValueError(f'unknown extractor {word_feature_extractor}!')
        # The linear layer that maps from hidden state space to tag space

    def forward(self, word_inputs, word_seq_lengths, char_inputs=None, char_seq_lengths=None):
        """
            input:
                word_inputs: (batch_size, sent_len)
                feature_inputs: [(batch_size, sent_len), ...] list of variables
                word_seq_lengths: list of batch_size, (batch_size,1)
                char_inputs: (batch_size*sent_len, word_length)
                char_seq_lengths: list of whole batch_size for char, (batch_size*sent_len, 1)
            output:
                Variable(batch_size, sent_len, hidden_dim)
        """
        word_represent = self.wordrep(word_inputs, char_inputs, char_seq_lengths)
        ## word_embs (batch_size, seq_len, embed_size)
        packed_words = pack_padded_sequence(word_represent, word_seq_lengths, batch_first=True, enforce_sorted=False)
        lstm_out, hidden = self.lstm(packed_words, None)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        ## lstm_out (seq_len, seq_len, hidden_size)
        feature_out = self.dropout(lstm_out)
        ## feature_out (batch_size, seq_len, hidden_size)
        return feature_out
