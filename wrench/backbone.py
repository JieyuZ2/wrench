from abc import abstractmethod
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from transformers import AutoModel, AutoConfig

from .layers import WordSequence


class BackBone(nn.Module):
    def __init__(self, n_class, binary_mode=False):
        if binary_mode:
            assert n_class == 2
            n_class = 1
        self.n_class = n_class
        super(BackBone, self).__init__()
        self.dummy_param = nn.Parameter(torch.empty(0))

    @property
    def device(self):
        return self.dummy_param.device

    def get_device(self):
        return self.dummy_param.device

    @abstractmethod
    def forward(self, batch: Dict, return_features: Optional[bool] = False):
        pass


class BERTBackBone(BackBone):
    def __init__(self, n_class, model_name='bert-base-cased', fine_tune_layers=-1, binary_mode=False):
        super(BERTBackBone, self).__init__(n_class=n_class, binary_mode=binary_mode)
        self.model_name = model_name
        self.config = AutoConfig.from_pretrained(model_name, num_labels=self.n_class, output_hidden_states=True)
        self.model = AutoModel.from_pretrained(model_name, config=self.config)

        if fine_tune_layers >= 0:
            for param in self.model.base_model.embeddings.parameters(): param.requires_grad = False
            if fine_tune_layers > 0:
                n_layers = len(self.model.base_model.encoder.layer)
                for layer in self.model.base_model.encoder.layer[:n_layers - fine_tune_layers]:
                    for param in layer.parameters():
                        param.requires_grad = False

    @abstractmethod
    def forward(self, batch: Dict, return_features: Optional[bool] = False):
        pass


""" backbone for classification """


class LogReg(BackBone):
    def __init__(self, n_class, input_size, binary_mode=False, **kwargs):
        super(LogReg, self).__init__(n_class=n_class, binary_mode=binary_mode)
        self.linear = nn.Linear(input_size, self.n_class)

    def forward(self, batch, return_features=False):
        x = batch['features'].to(self.get_device())
        x = self.linear(x)
        return x


class MLP(BackBone):
    def __init__(self, n_class, input_size, n_hidden_layers=1, hidden_size=100, dropout=0.0, binary_mode=False, **kwargs):
        super(MLP, self).__init__(n_class=n_class, binary_mode=binary_mode)
        layers = [nn.Linear(input_size, hidden_size), nn.ReLU(), nn.Dropout(p=dropout)]
        for i in range(n_hidden_layers - 1):
            layers.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Dropout(p=dropout)])
        self.fcs = nn.Sequential(*layers)
        self.last_layer = nn.Linear(hidden_size, self.n_class)
        self.hidden_size = hidden_size

    def forward(self, batch, return_features=False):
        x = batch['features'].to(self.get_device())
        h = self.fcs(x)
        logits = self.last_layer(h)
        if return_features:
            return logits, h
        else:
            return logits


""" torchvision for image classification """


class ImageClassifier(BackBone):

    def __init__(self, n_class, model_name='resnet18', binary_mode=False, **kwargs):
        super(ImageClassifier, self).__init__(n_class=n_class, binary_mode=binary_mode)

        pretrained_model = getattr(torchvision.models, model_name)(pretrained=False)
        self.model = nn.Sequential(*list(pretrained_model.children())[:-1])

        # pretrained_model = getattr(torchvision.models, model_name)(pretrained=load_pretrained)
        # self.model = nn.Sequential(*list(pretrained_model.children())[:-1])
        # if load_pretrained and (not finetune_pretrained):
        #     for param in self.model.parameters():
        #         param.requires_grad = False

        self.hidden_size = pretrained_model.fc.in_features
        self.fc = nn.Linear(self.hidden_size, n_class)

    def forward(self, batch, return_features=False):
        h = self.model(batch['image'].to(self.get_device()))
        h = torch.flatten(h, 1)
        logits = self.fc(h)
        if return_features:
            return logits, h
        else:
            return logits


""" BERT for text classification """


#######################################################################################################################
class BertTextClassifier(BERTBackBone):
    """
    Bert with a MLP on top for text classification
    """

    def __init__(self, n_class, model_name='bert-base-cased', fine_tune_layers=-1, max_tokens=512, binary_mode=False, **kwargs):
        super(BertTextClassifier, self).__init__(n_class=n_class, model_name=model_name, fine_tune_layers=fine_tune_layers, binary_mode=binary_mode)

        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size, self.config.num_labels)
        self.max_tokens = max_tokens
        self.hidden_size = self.config.hidden_size

    def forward(self, batch, return_features=False):  # inputs: [batch, t]
        device = self.get_device()
        outputs = self.model(input_ids=batch["input_ids"].to(device), attention_mask=batch['mask'].to(device))
        h = self.dropout(outputs.pooler_output)
        output = self.classifier(h)
        if return_features:
            return output, h
        else:
            return output


""" BERT for relation classification """


#######################################################################################################################
class FClayer(nn.Module):
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


class BertRelationClassifier(BERTBackBone):
    """
    BERT with a MLP on top for relation classification
    """

    def __init__(self, n_class, model_name='bert-base-cased', fine_tune_layers=-1, binary_mode=False, **kwargs):
        super(BertRelationClassifier, self).__init__(n_class=n_class, model_name=model_name, fine_tune_layers=fine_tune_layers, binary_mode=binary_mode)
        self.fc_cls = FClayer(self.config.hidden_size, self.config.hidden_size, dropout=self.config.hidden_dropout_prob)
        self.fc_e1 = FClayer(self.config.hidden_size, self.config.hidden_size, dropout=self.config.hidden_dropout_prob)
        self.fc_e2 = FClayer(self.config.hidden_size, self.config.hidden_size, dropout=self.config.hidden_dropout_prob)
        self.output = FClayer(self.config.hidden_size * 3, self.n_class, dropout=self.config.hidden_dropout_prob, activation=False)
        self.hidden_size = self.config.hidden_size * 3

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

        sum_vector = torch.bmm(e_mask_unsqueeze.float(), hidden_output).squeeze(1)  # [b, 1, j-i+1] * [b, j-i+1, dim] = [b, 1, dim] -> [b, dim]
        avg_vector = sum_vector.float() / length_tensor.float()  # broadcasting
        return avg_vector

    def forward(self, batch, return_features=False):
        device = self.get_device()
        outputs = self.model(input_ids=batch["input_ids"].to(device), attention_mask=batch['mask'].to(device))
        bert_out = outputs.last_hidden_state
        cls_embs = self.fc_cls(outputs.pooler_output)
        ent1_avg = self.fc_e1(self.entity_average(bert_out, batch['e1_mask'].to(device)))
        ent2_avg = self.fc_e2(self.entity_average(bert_out, batch['e2_mask'].to(device)))
        h = torch.cat([cls_embs, ent1_avg, ent2_avg], dim=-1)
        output = self.output(h)
        if return_features:
            return output, h
        else:
            return output


""" for sequence tagging """


class CRFTagger(BackBone):
    def __init__(self, n_class, use_crf):
        super(CRFTagger, self).__init__(n_class=n_class)
        self.use_crf = use_crf
        if self.use_crf:
            self.crf = CRF(n_class)

    def calculate_loss(self, batch, batch_label):
        device = self.get_device()
        outs = self.get_features(batch)

        mask = batch['mask'].to(device)
        batch_size, seq_len, _ = outs.shape
        batch_label = batch_label[:, :seq_len].to(device)

        if self.use_crf:
            total_loss = self.crf.neg_log_likelihood_loss(outs, mask, batch_label)
            total_loss = total_loss / batch_size
        else:
            outs = outs.view(batch_size * seq_len, -1)
            mask = mask.reshape(batch_size * seq_len).bool()
            batch_label = batch_label.reshape(batch_size * seq_len)
            score = F.log_softmax(outs, 1)
            total_loss = F.nll_loss(score[mask], batch_label[mask])
        return total_loss

    def forward(self, batch):
        device = self.get_device()
        outs = self.get_features(batch)

        mask = batch['mask'].to(device)
        if self.use_crf:
            scores, tag_seq = self.crf(outs, mask)
        else:
            batch_size, seq_len, _ = outs.shape
            outs = outs.view(batch_size * seq_len, -1)
            _, tag = torch.max(outs, 1)
            tag = tag.view(batch_size, seq_len)
            tag_seq = [[tt for tt, mm in zip(t, m) if mm] for t, m in zip(tag.tolist(), mask.tolist())]

        return tag_seq

    @abstractmethod
    def get_features(self, batch):
        pass


class LSTMSeqTagger(CRFTagger):
    def __init__(self,
                 n_class,
                 word_vocab_size,
                 char_vocab_size,
                 use_crf,
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
                 char_feature_extractor,
                 **kwargs):
        super(LSTMSeqTagger, self).__init__(n_class=n_class, use_crf=use_crf)
        if use_crf:
            n_class += 2
        self.word_hidden = WordSequence(
            word_vocab_size=word_vocab_size,
            char_vocab_size=char_vocab_size,
            dropout=dropout,
            word_embedding=word_embedding,
            word_emb_dim=word_emb_dim,
            word_hidden_dim=word_hidden_dim,
            word_feature_extractor=word_feature_extractor,
            n_word_hidden_layer=n_word_hidden_layer,
            use_char=use_char,
            char_embedding=char_embedding,
            char_emb_dim=char_emb_dim,
            char_hidden_dim=char_hidden_dim,
            char_feature_extractor=char_feature_extractor
        )
        self.classifier = nn.Linear(word_hidden_dim, n_class)

    def get_features(self, batch):
        device = self.get_device()
        word_inputs = batch['word'].to(device)
        word_seq_lengths = batch['word_length']
        char_inputs = batch['char'].to(device)
        char_seq_lengths = batch['char_length']
        char_inputs = char_inputs.flatten(0, 1)
        char_seq_lengths = char_seq_lengths.flatten()
        feature_out = self.word_hidden(word_inputs, word_seq_lengths, char_inputs, char_seq_lengths)
        outs = self.classifier(feature_out)
        return outs


class BertSeqTagger(CRFTagger):
    """
    BERT for sequence tagging
    """

    def __init__(self, n_class, model_name='bert-base-cased', fine_tune_layers=-1, use_crf=True, **kwargs):
        super(BertSeqTagger, self).__init__(n_class=n_class, use_crf=use_crf)
        self.model_name = model_name
        config = AutoConfig.from_pretrained(self.model_name, output_hidden_states=True)
        self.model = AutoModel.from_pretrained(self.model_name, config=config)
        self.config = config

        if fine_tune_layers >= 0:
            for param in self.model.base_model.embeddings.parameters(): param.requires_grad = False
            if fine_tune_layers > 0:
                n_layers = len(self.model.base_model.encoder.layer)
                for layer in self.model.base_model.encoder.layer[:n_layers - fine_tune_layers]:
                    for param in layer.parameters():
                        param.requires_grad = False

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.use_crf = use_crf
        if self.use_crf:
            self.classifier = nn.Linear(config.hidden_size, n_class + 2)  # consider <START> and <END> token
        else:
            self.classifier = nn.Linear(config.hidden_size, n_class + 1)

    def get_features(self, batch):
        device = self.get_device()
        outputs = self.model(input_ids=batch["input_ids"].to(device), attention_mask=batch['attention_mask'].to(device))
        outs = self.classifier(self.dropout(outputs.last_hidden_state))
        if self.use_crf:
            return outs
        else:
            return outs[:, :, :-1]


START_TAG = -2
STOP_TAG = -1


class CRF(BackBone):

    def __init__(self, n_class, batch_mode=True):
        super(CRF, self).__init__(n_class=n_class)
        # Matrix of transition parameters.  Entry i,j is the score of transitioning from i to j.
        self.n_class = n_class + 2
        self.batch_mode = batch_mode
        # # We add 2 here, because of START_TAG and STOP_TAG
        # # transitions (f_tag_size, t_tag_size), transition value from f_tag to t_tag
        init_transitions = torch.randn(self.n_class, self.n_class)
        self.START_TAG = -2
        self.STOP_TAG = -1
        init_transitions[:, self.START_TAG] = -1e5
        init_transitions[self.STOP_TAG, :] = -1e5
        self.transitions = nn.Parameter(init_transitions, requires_grad=True)
        self.start_id = nn.Parameter(torch.LongTensor([self.START_TAG]), requires_grad=False)
        self.stop_id = nn.Parameter(torch.LongTensor([self.STOP_TAG]), requires_grad=False)

    def _score_sentence_batch(self, feats, tags, mask, transitions=None):
        # Gives the score of a provided tag sequence
        # tags is ground_truth, a list of ints, length is len(sentence)
        # feats is a 2D tensor, len(sentence) * n_class
        if transitions is None:
            transitions = self.transitions

        batch_size = tags.size(0)
        seq_len = mask.long().sum(1)
        r_batch = torch.arange(batch_size)

        pad_start_tags = torch.cat([self.start_id.expand(batch_size, 1), tags], -1)
        pad_stop_tags = torch.cat([tags, self.stop_id.expand(batch_size, 1)], -1)
        pad_stop_tags[r_batch, seq_len] = self.stop_id
        t = transitions[pad_start_tags, pad_stop_tags]
        t_score = torch.sum(t.cumsum(1)[r_batch, seq_len])

        f_score = torch.sum(torch.gather(feats, -1, tags.unsqueeze(2)).squeeze(2).masked_select(mask.bool()))

        score = t_score + f_score
        return score

    def _score_sentence(self, feats, tags, transitions=None):
        # Gives the score of a provided tag sequence
        # tags is ground_truth, a list of ints, length is len(sentence)
        # feats is a 2D tensor, len(sentence) * n_class

        if transitions is None:
            transitions = self.transitions

        pad_start_tags = torch.cat([self.start_id, tags])
        pad_stop_tags = torch.cat([tags, self.stop_id])

        r = torch.arange(feats.size(0))
        score = torch.sum(transitions[pad_start_tags, pad_stop_tags]) + torch.sum(feats[r, tags])
        return score

    def _forward_alg_batch(self, feats, mask, transitions=None):
        # calculate in log domain
        # feats is len(sentence) * n_class
        if transitions is None:
            transitions = self.transitions

        device = self.get_device()
        batch_size, max_seq_len, target_size = feats.shape
        alpha = torch.full((batch_size, 1, target_size), -10000.0, device=device)
        alpha[:, 0, self.START_TAG] = 0.0
        mask = mask.bool()

        for i in range(max_seq_len):
            feat = feats[:, i, :]
            mask_i = mask[:, i]
            alpha = torch.where(mask_i.view(-1, 1, 1), torch.logsumexp(alpha.transpose(1, 2) + feat.unsqueeze(1) + transitions, dim=1, keepdim=True), alpha)

        last = torch.logsumexp(alpha.transpose(1, 2) + 0 + transitions[:, [self.STOP_TAG]], dim=1)
        score = torch.sum(last)
        return score

    def _forward_alg(self, feats, transitions=None):
        # calculate in log domain
        # feats is len(sentence) * n_class
        if transitions is None:
            transitions = self.transitions

        device = self.get_device()
        alpha = torch.full((1, self.n_class), -10000.0, device=device)
        alpha[0][self.START_TAG] = 0.0
        for feat in feats:
            alpha = torch.logsumexp(alpha.T + feat.unsqueeze(0) + transitions, dim=0, keepdim=True)
        return torch.logsumexp(alpha.T + 0 + transitions[:, [self.STOP_TAG]], dim=0)[0]

    def viterbi_decode_batch(self, feats, mask, transitions=None):
        if transitions is None:
            transitions = self.transitions

        device = self.get_device()
        batch_size, max_seq_len, target_size = feats.shape
        backtrace = torch.zeros((batch_size, max_seq_len, target_size)).long()
        alpha = torch.full((batch_size, 1, target_size), -10000.0, device=device)
        alpha[:, 0, self.START_TAG] = 0.0
        mask = mask.bool()

        for i in range(max_seq_len):
            feat = feats[:, i, :]
            mask_i = mask[:, i]
            smat = (alpha.transpose(1, 2) + feat.unsqueeze(1) + transitions)  # (n_class, n_class)
            alpha = torch.where(mask_i.view(-1, 1, 1), torch.logsumexp(smat, dim=1, keepdim=True), alpha)
            backtrace[:, i, :] = smat.argmax(1)
        # backtrack
        smat = alpha.transpose(1, 2) + 0 + transitions[:, [self.STOP_TAG]]
        best_tag_ids = smat.argmax(1).long()

        seq_len = mask.long().sum(1)
        best_paths = []
        for backtrace_i, best_tag_id, l in zip(backtrace, best_tag_ids, seq_len):
            best_path = [best_tag_id.item()]
            for bptrs_t in reversed(backtrace_i[1:l]):  # ignore START_TAG
                best_tag_id = bptrs_t[best_tag_id].item()
                best_path.append(best_tag_id)
            best_paths.append(best_path[::-1])
        return torch.logsumexp(smat, dim=1).squeeze().tolist(), best_paths

    def viterbi_decode(self, feats, transitions=None):
        if transitions is None:
            transitions = self.transitions

        device = self.get_device()
        backtrace = []
        alpha = torch.full((1, self.n_class), -10000.0, device=device)
        alpha[0][self.START_TAG] = 0
        for feat in feats:
            smat = (alpha.T + feat.unsqueeze(0) + transitions)  # (n_class, n_class)
            backtrace.append(smat.argmax(0))  # column_max
            alpha = torch.logsumexp(smat, dim=0, keepdim=True)
        # backtrack
        smat = alpha.T + 0 + transitions[:, [self.STOP_TAG]]
        best_tag_id = smat.flatten().argmax().item()
        best_path = [best_tag_id]
        for bptrs_t in reversed(backtrace[1:]):  # ignore START_TAG
            best_tag_id = bptrs_t[best_tag_id].item()
            best_path.append(best_tag_id)
        return torch.logsumexp(smat, dim=0).item(), best_path[::-1]

    def neg_log_likelihood_loss(self, feats, mask, tags, transitions=None):
        # sentence, tags is a list of ints
        # features is a 2D tensor, len(sentence) * self.n_class
        if self.batch_mode:
            nll_loss = self._forward_alg_batch(feats, mask, transitions) - self._score_sentence_batch(feats, tags, mask, transitions)
        else:
            nll_loss = 0.0
            batch_size = len(feats)
            for i in range(batch_size):
                length = mask[i].long().sum()
                feat_i = feats[i][:length]
                tags_i = tags[i][:length]
                forward_score = self._forward_alg(feat_i, transitions)
                gold_score = self._score_sentence(feat_i, tags_i, transitions)
                nll_loss += forward_score - gold_score
        return nll_loss

    def forward(self, feats, mask):
        # viterbi to get tag_seq
        if self.batch_mode:
            score, tags = self.viterbi_decode_batch(feats, mask)
        else:
            tags = []
            scores = []
            batch_size = len(feats)
            for i in range(batch_size):
                length = mask[i].long().sum()
                feat_i = feats[i][:length]
                score, tag_seq = self.viterbi_decode(feat_i)
                tags.append(tag_seq)
                scores.append(score)
        return score, tags


class MultiCRF(CRF):

    def __init__(self, n_class, n_source, batch_mode=True):
        super(MultiCRF, self).__init__(n_class=n_class, batch_mode=batch_mode)
        self.n_source = n_source
        init_transitions = torch.randn(n_source, self.n_class, self.n_class)
        init_transitions[:, :, self.START_TAG] = -1e5
        init_transitions[:, self.STOP_TAG, :] = -1e5
        self.transitions = nn.Parameter(init_transitions, requires_grad=True)

    def neg_log_likelihood_loss(self, feats, mask, tags, idx=None, attn_weight=None):
        if attn_weight is None:
            assert idx is not None
            transitions = self.transitions[idx]
            return super().neg_log_likelihood_loss(feats, mask, tags, transitions)
        else:
            assert attn_weight is not None, 'weight should not be None in Phase 2!'
            transitions_l = torch.tensordot(attn_weight, self.transitions, dims=([1], [0]))

            nll_loss = self._forward_alg_batch_w_transitions(feats, mask, transitions_l) - \
                       self._score_sentence_w_transitions(feats, tags, mask, transitions_l)
            return nll_loss

    def _score_sentence_w_transitions(self, feats, tags, mask, transitions):
        batch_size = tags.size(0)
        seq_len = mask.long().sum(1)
        r_batch = torch.arange(batch_size)

        pad_start_tags = torch.cat([self.start_id.expand(batch_size, 1), tags], -1)
        pad_stop_tags = torch.cat([tags, self.stop_id.expand(batch_size, 1)], -1)
        pad_stop_tags[r_batch, seq_len] = self.stop_id
        t = transitions[r_batch.view(-1, 1), pad_start_tags, pad_stop_tags]
        t_score = torch.sum(t.cumsum(1)[r_batch, seq_len])

        f_score = torch.sum(torch.gather(feats, -1, tags.unsqueeze(2)).squeeze(2).masked_select(mask.bool()))

        score = t_score + f_score
        return score

    def _forward_alg_batch_w_transitions(self, feats, mask, transitions):
        device = self.get_device()
        batch_size, max_seq_len, target_size = feats.shape
        alpha = torch.full((batch_size, 1, target_size), -10000.0, device=device)
        alpha[:, 0, self.START_TAG] = 0.0
        mask = mask.bool()

        for i in range(max_seq_len):
            feat = feats[:, i, :]
            mask_i = mask[:, i]
            alpha = torch.where(mask_i.view(-1, 1, 1), torch.logsumexp(alpha.transpose(1, 2) + feat.unsqueeze(1) + transitions, dim=1, keepdim=True), alpha)

        last = torch.logsumexp(alpha.transpose(1, 2) + 0 + transitions[:, :, [self.STOP_TAG]], dim=1)
        score = torch.sum(last)
        return score

    def viterbi_decode_w_transitions(self, feats, mask, transitions):
        device = self.get_device()
        batch_size, max_seq_len, target_size = feats.shape
        backtrace = torch.zeros((batch_size, max_seq_len, target_size)).long()
        alpha = torch.full((batch_size, 1, target_size), -10000.0, device=device)
        alpha[:, 0, self.START_TAG] = 0.0
        mask = mask.bool()

        for i in range(max_seq_len):
            feat = feats[:, i, :]
            mask_i = mask[:, i]
            smat = (alpha.transpose(1, 2) + feat.unsqueeze(1) + transitions)  # (n_class, n_class)
            alpha = torch.where(mask_i.view(-1, 1, 1), torch.logsumexp(smat, dim=1, keepdim=True), alpha)
            backtrace[:, i, :] = smat.argmax(1)
        # backtrack
        smat = alpha.transpose(1, 2) + 0 + transitions[:, :, [self.STOP_TAG]]
        best_tag_ids = smat.argmax(1).long()

        seq_len = mask.long().sum(1)
        best_paths = []
        for backtrace_i, best_tag_id, l in zip(backtrace, best_tag_ids, seq_len):
            best_path = [best_tag_id.item()]
            for bptrs_t in reversed(backtrace_i[1:l]):  # ignore START_TAG
                best_tag_id = bptrs_t[best_tag_id].item()
                best_path.append(best_tag_id)
            best_paths.append(best_path[::-1])
        return torch.logsumexp(smat, dim=1).squeeze().tolist(), best_paths

    def forward(self, feats, mask, attn_weight):
        transitions_l = torch.tensordot(attn_weight, self.transitions, dims=([1], [0]))
        return self.viterbi_decode_w_transitions(feats, mask, transitions_l)
