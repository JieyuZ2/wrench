from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel, AutoConfig



class BackBone(nn.Module, ABC):
    def __init__(self, n_class, binary_mode=False):
        if binary_mode:
            assert n_class == 2
            n_class = 1
        self.n_class = n_class
        super(BackBone, self).__init__()
        self.dummy_param = nn.Parameter(torch.empty(0))

    def get_device(self):
        return self.dummy_param.device

    @abstractmethod
    def forward(self, batch: Dict, return_features: Optional[bool] = False):
        pass



class LogReg(BackBone):
    def __init__(self, n_class, input_size, binary_mode=False, **kwargs):
        super(LogReg, self).__init__(n_class=n_class, binary_mode=binary_mode)
        self.linear = nn.Linear(input_size, self.n_class)

    def forward(self, batch, return_features=False):
        x = batch['features'].to(self.get_device())
        x = self.linear(x)
        return x



class MLP(BackBone):
    def __init__(self, n_class, input_size, hidden_size=100, dropout=0.0, binary_mode=False, **kwargs):
        super(MLP, self).__init__(n_class=n_class, binary_mode=binary_mode)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, self.n_class)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, batch, return_features=False):
        x = batch['features'].to(self.get_device())
        h = F.relu(self.dropout(self.fc1(x)))
        output = self.fc2(h)
        if return_features:
            return output, h
        else:
            return output


""" FC Layer"""
#######################################################################################################################
class FClayer(nn.Module):
    """
    MLP layer for classification
    """

    def __init__(self, input_dim, hidden_size=100, dropout=0., activation=True):
        super(FClayer, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(input_dim, hidden_size)
        self.tanh = nn.Tanh()
        self.activation = activation

    def forward(self, x):
        x = self.dropout(x)
        x = self.linear(x)
        if self.activation:
            return self.tanh(x)
        else:
            return x


""" BERT for text classification """
#######################################################################################################################
class BertTextClassifier(BackBone):
    """
    Bert with a MLP on top for text classification
    """

    def __init__(self, n_class, model_name='bert-base-cased', fine_tune_layers=-1, max_length=512, binary_mode=False, **kwargs):
        super(BertTextClassifier, self).__init__(n_class=n_class, binary_mode=binary_mode)
        config = AutoConfig.from_pretrained(model_name, num_labels=self.n_class, output_hidden_states=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, config=config)
        self.config = config

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        if fine_tune_layers >= 0:
            for param in self.model.base_model.embeddings.parameters(): param.requires_grad = False
            if fine_tune_layers > 0:
                n_layers = len(self.model.base_model.encoder.layer)
                for layer in self.model.base_model.encoder.layer[:n_layers-fine_tune_layers]:
                    for param in layer.parameters():
                        param.requires_grad = False

        self.model_name = model_name
        self.max_length = max_length

    def forward(self, batch, return_features=False):  # inputs: [batch, t]
        inputs = self.tokenizer(batch['data']['text'], padding=True, return_tensors='pt', max_length=self.max_length, truncation=True)
        inputs = {k:v.to(self.get_device()) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        h = self.dropout(outputs.pooler_output)
        output = self.classifier(h)
        if return_features:
            return output, h
        else:
            return output


""" BERT for relation classification """
#######################################################################################################################
class BertRelationClassifier(BackBone):
    """
    BERT with a MLP on top for relation classification
    """
    def __init__(self, n_class, model_name='bert-base-cased', fine_tune_layers=-1, binary_mode=False, **kwargs):
        super(BertRelationClassifier, self).__init__(n_class=n_class, binary_mode=binary_mode)
        config = AutoConfig.from_pretrained(model_name, output_hidden_states=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, config=config)
        self.config = config

        if fine_tune_layers >= 0:
            for param in self.model.base_model.embeddings.parameters(): param.requires_grad = False
            if fine_tune_layers > 0:
                n_layers = len(self.model.base_model.encoder.layer)
                for layer in self.model.base_model.encoder.layer[:n_layers - fine_tune_layers]:
                    for param in layer.parameters():
                        param.requires_grad = False

        self.fc_cls = FClayer(config.hidden_size, config.hidden_size, dropout=config.hidden_dropout_prob)
        self.fc_e1 = FClayer(config.hidden_size, config.hidden_size, dropout=config.hidden_dropout_prob)
        self.fc_e2 = FClayer(config.hidden_size, config.hidden_size, dropout=config.hidden_dropout_prob)
        self.output = FClayer(config.hidden_size * 3, self.n_class, dropout=config.hidden_dropout_prob, activation=False)
        self.model_name = model_name

    @staticmethod
    def entity_average(hidden_output, e_mask):
        """
        Average the entity hidden state vectors (H_i ~ H_j)
        :param hidden_output: [batch_size, j-i+1, dim]
        :param e_mask: [batch_size, max_seq_len]
                e.g. e_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
        :return: [batch_size, dim]
        """

        e_mask_unsqueeze = e_mask.unsqueeze(1)  # [b, 1, j-i+1]
        length_tensor = (e_mask != 0).sum(dim=1).unsqueeze(1)  # [batch_size, 1]

        sum_vector = torch.bmm(e_mask_unsqueeze.float(), hidden_output).squeeze(1) # [b, 1, j-i+1] * [b, j-i+1, dim] = [b, 1, dim] -> [b, dim]
        avg_vector = sum_vector.float() / length_tensor.float()  # broadcasting
        return avg_vector

    def forward(self, batch, return_features=False):
        device = self.get_device()
        input_ids, e1_mask, e2_mask = self.preprocess(batch['data'])
        input_ids = input_ids.to(device)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)  # [t, batch, hidden]
        bert_out = outputs.last_hidden_state
        cls_embs = self.fc_cls(outputs.pooler_output)
        ent1_avg = self.fc_e1(self.entity_average(bert_out, e1_mask.to(device)))
        ent2_avg = self.fc_e2(self.entity_average(bert_out, e2_mask.to(device)))
        h = torch.cat([cls_embs, ent1_avg, ent2_avg], dim=-1)
        output = self.output(h)
        if return_features:
            return output, h
        else:
            return output

    def preprocess(self, batch):
        span1_l = batch.pop('span1')
        span2_l = batch.pop('span2')
        batch_l = [dict(zip(batch, t)) for t in zip(*batch.values())]
        batch['span1'] = span1_l
        batch['span2'] = span2_l
        tokens_l, e1s_l, e1n_l, e2s_l, e2n_l = [], [], [], [], []
        for i, item in enumerate(batch_l):
            sentence = item['text']

            span1s, span1n, span2s, span2n = span1_l[0][i], span1_l[1][i], span2_l[0][i], span2_l[1][i]

            e1_first =  span1s < span2s
            if e1_first:
                left_text = sentence[:span1s]
                between_text = sentence[span1n:span2s]
                right_text = sentence[span2n:]
            else:
                left_text = sentence[:span2s]
                between_text = sentence[span2n:span1s]
                right_text = sentence[span1n:]
            left_tkns = self.tokenizer.tokenize(left_text)
            between_tkns = self.tokenizer.tokenize(between_text)
            right_tkns = self.tokenizer.tokenize(right_text)
            e1_tkns = self.tokenizer.tokenize(item['entity1'])
            e2_tkns = self.tokenizer.tokenize(item['entity2'])

            if e1_first:
                tokens = ["[CLS]"] + left_tkns + ["$"] + e1_tkns + ["$"] + between_tkns + ["#"] + e2_tkns + ["#"] + right_tkns + ["[SEP]"]
                e1s = len(left_tkns) + 1  # inclusive
                e1n = e1s + len(e1_tkns) + 2  # exclusive
                e2s = e1n + len(between_tkns)
                e2n = e2s + len(e2_tkns) + 2
                end = e2n
            else:
                tokens = ["[CLS]"] + left_tkns + ["#"] + e2_tkns + ["#"] + between_tkns + ["$"] + e1_tkns + ["$"] + right_tkns + ["[SEP]"]
                e2s = len(left_tkns) + 1  # inclusive
                e2n = e2s + len(e2_tkns) + 2  # exclusive
                e1s = e2n + len(between_tkns)
                e1n = e1s + len(e1_tkns) + 2
                end = e1n

            if len(tokens) > 512:
                if end >= 512:
                    len_truncated = len(between_tkns) + len(e1_tkns) + len(e2_tkns) + 6
                    if len_truncated > 512:
                        diff = len_truncated - 512
                        len_between = len(between_tkns)
                        between_tkns = between_tkns[:(len_between - diff) // 2] + between_tkns[(len_between - diff) // 2 + diff:]
                    if e1_first:
                        truncated = ["[CLS]"] + ["$"] + e1_tkns + ["$"] + between_tkns + ["#"] + e2_tkns + ["#"] + ["[SEP]"]
                        e1s = 1  # inclusive
                        e1n = e1s + len(e1_tkns) + 2  # exclusive
                        e2s = e1n + len(between_tkns)
                        e2n = e2s + len(e2_tkns) + 2
                    else:
                        truncated = ["[CLS]"] + ["#"] + e2_tkns + ["#"] + between_tkns + ["$"] + e1_tkns + ["$"] + ["[SEP]"]
                        e2s = 1  # inclusive
                        e2n = e2s + len(e2_tkns) + 2  # exclusive
                        e1s = e2n + len(between_tkns)
                        e1n = e1s + len(e1_tkns) + 2
                    tokens = truncated
                    assert len(tokens) <= 512
                else:
                    tokens = tokens[:512]

            assert e1_tkns == tokens[e1s + 1:e1n - 1]
            assert e2_tkns == tokens[e2s + 1:e2n - 1]

            e1s_l.append(e1s)
            e1n_l.append(e1n)
            e2s_l.append(e2s)
            e2n_l.append(e2n)
            tokens_l.append(self.tokenizer.convert_tokens_to_ids(tokens))

        max_len = max(list(map(len, tokens_l)))
        input_ids = torch.LongTensor([t + [self.tokenizer.pad_token_id] * (max_len - len(t)) for t in tokens_l])
        e1_mask = torch.zeros_like(input_ids)
        e2_mask = torch.zeros_like(input_ids)
        for i in range(len(batch_l)):
            e1_mask[i, e1s_l[i]:e1n_l[i]] = 1
            e2_mask[i, e2s_l[i]:e2n_l[i]] = 1
        return input_ids, e1_mask, e2_mask


""" BERT for entity classification """
#######################################################################################################################
class BertEntityClassifier(BackBone):
    """
    BERT with a MLP on top for relation classification
    """
    def __init__(self, n_class, model_name='bert-base-cased', fine_tune_layers=-1, binary_mode=False, **kwargs):
        super(BertEntityClassifier, self).__init__(n_class=n_class, binary_mode=binary_mode)
        config = AutoConfig.from_pretrained(model_name, output_hidden_states=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, config=config)
        self.config = config
        self.n_class = n_class

        if fine_tune_layers >= 0:
            for param in self.model.base_model.embeddings.parameters(): param.requires_grad = False
            if fine_tune_layers > 0:
                n_layers = len(self.model.base_model.encoder.layer)
                for layer in self.model.base_model.encoder.layer[:n_layers - fine_tune_layers]:
                    for param in layer.parameters():
                        param.requires_grad = False

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, n_class)
        self.model_name = model_name

    @staticmethod
    def entity_average(hidden_output, e_mask):
        """
        Average the entity hidden state vectors (H_i ~ H_j)
        :param hidden_output: [batch_size, j-i+1, dim]
        :param e_mask: [batch_size, max_seq_len]
                e.g. e_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
        :return: [batch_size, dim]
        """

        e_mask_unsqueeze = e_mask.unsqueeze(1)  # [b, 1, j-i+1]
        length_tensor = (e_mask != 0).sum(dim=1).unsqueeze(1)  # [batch_size, 1]

        sum_vector = torch.bmm(e_mask_unsqueeze.float(), hidden_output).squeeze(1) # [b, 1, j-i+1] * [b, j-i+1, dim] = [b, 1, dim] -> [b, dim]
        avg_vector = sum_vector.float() / length_tensor.float()  # broadcasting
        return avg_vector

    def forward(self, batch, return_features=False):
        input_ids, e1_mask = self.preprocess(batch['data'])
        outputs = self.model(input_ids=input_ids)  # [t, batch, hidden]
        bert_out = outputs.last_hidden_state
        h = self.dropout(self.entity_average(bert_out, e1_mask))
        output = self.classifier(h)
        if return_features:
            return output, h
        else:
            return output

    def preprocess(self, batch):
        span_s_l, span_e_l = batch.pop('span')
        e_l, s_l, tokens_l = [], [], []
        for i, sentence in enumerate(batch['text']):
            left_tkns = ["[CLS]"] + self.tokenizer.tokenize(sentence[:span_s_l[i]])
            entity_tkns = self.tokenizer.tokenize(sentence[span_s_l[i]:span_e_l[i]])
            right_tkns = self.tokenizer.tokenize(sentence[span_e_l[i]:]) + ["[SEP]"]
            tokens = left_tkns + entity_tkns + right_tkns

            e = len(left_tkns+entity_tkns)
            s = len(left_tkns)
            e_l.append(e)
            s_l.append(s)
            tokens_l.append(self.tokenizer.convert_tokens_to_ids(tokens))

        max_len = max(list(map(len, tokens_l)))
        input_ids = torch.LongTensor([t + [self.tokenizer.pad_token_id] * (max_len - len(t)) for t in tokens_l])
        e1_mask = torch.zeros_like(input_ids)
        for i in range(len(tokens_l)):
            e1_mask[i, s_l[i]:e_l[i]] = 1
        device = self.get_device()
        return input_ids.to(device), e1_mask.to(device)

