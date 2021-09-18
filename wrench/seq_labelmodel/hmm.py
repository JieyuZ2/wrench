import logging
import re
from typing import Any, List, Optional, Union

import numpy as np
from skweak.aggregation import HMM as HMM_
from spacy.lang.en import English
from spacy.tokenizer import Tokenizer
from spacy.tokens import Span

from ..basemodel import BaseSeqModel
from ..dataset import BaseSeqDataset
from ..utils import set_seed

logger = logging.getLogger(__name__)

ABSTAIN = -1


def label_to_span(labels: List[str],
                  scheme: Optional[str] = 'BIO') -> dict:
    """
    convert labels to spans
    :param labels: a list of labels
    :param scheme: labeling scheme, in ['BIO', 'BILOU'].
    :return: labeled spans, a list of tuples (start_idx, end_idx, label)
    """
    assert scheme in ['BIO', 'BILOU'], ValueError("unknown labeling scheme")

    labeled_spans = dict()
    i = 0
    while i < len(labels):
        if labels[i] == 'O':
            i += 1
            continue
        else:
            if scheme == 'BIO':
                if labels[i][0] == 'B':
                    start = i
                    lb = labels[i][2:]
                    i += 1
                    try:
                        while labels[i][0] == 'I':
                            i += 1
                        end = i
                        labeled_spans[(start, end)] = lb
                    except IndexError:
                        end = i
                        labeled_spans[(start, end)] = lb
                        i += 1
                # this should not happen
                elif labels[i][0] == 'I':
                    i += 1
            elif scheme == 'BILOU':
                if labels[i][0] == 'U':
                    start = i
                    end = i + 1
                    lb = labels[i][2:]
                    labeled_spans[(start, end)] = lb
                    i += 1
                elif labels[i][0] == 'B':
                    start = i
                    lb = labels[i][2:]
                    i += 1
                    try:
                        while labels[i][0] != 'L':
                            i += 1
                        end = i
                        labeled_spans[(start, end)] = lb
                    except IndexError:
                        end = i
                        labeled_spans[(start, end)] = lb
                        break
                    i += 1
                else:
                    i += 1

    return labeled_spans


class HMM(BaseSeqModel):
    def __init__(self,
                 n_epochs: Optional[int] = 50,
                 redundancy_factor: Optional[float] = 0.0,
                 **kwargs: Any):
        super().__init__()
        self.hyperparas = {
            "n_epochs"         : n_epochs,
            "redundancy_factor": redundancy_factor,
        }
        self.model = None

    def prepare_doc(self, corpus, weak_labels):
        nlp = English()
        nlp.tokenizer = Tokenizer(nlp.vocab, token_match=re.compile(r'\S').match)
        docs = []
        for text, weak_labels_i in zip(corpus, weak_labels):
            doc = nlp(' '.join(text))
            assert len(doc) == len(text)
            for i in range(self.n_lf):
                doc.spans[str(i)] = []
                weak_label = [self.id2label[ii] for ii in weak_labels_i[:, i]]
                for (start, end), label in label_to_span(weak_label).items():
                    span = Span(doc, start, end, label)
                    doc.spans[str(i)].append(span)
            docs.append(doc)
        return docs

    def fit(self,
            dataset_train: Union[BaseSeqDataset],
            verbose: Optional[bool] = False,
            seed: int = None,
            **kwargs: Any):

        if not verbose:
            logger.setLevel(logging.ERROR)

        seed = seed or np.random.randint(1e6)
        set_seed(seed)

        self._update_hyperparas(**kwargs)
        self.entity_types = dataset_train.entity_types
        self.id2label = dataset_train.id2label
        self.n_lf = dataset_train.n_lf

        docs = self.prepare_doc([item['text'] for item in dataset_train.examples], dataset_train.weak_labels)

        # with NoStdStreams(logger):
        hmm = HMM_("hmm", self.entity_types, redundancy_factor=self.hyperparas['redundancy_factor'])
        hmm.fit(docs, n_iter=self.hyperparas['n_epochs'])

        self.model = hmm

    def predict(self, dataset: BaseSeqDataset, **kwargs: Any):
        model = self.model

        docs = self.prepare_doc([item['text'] for item in dataset.examples], dataset.weak_labels)

        preds = []
        for doc in docs:
            sources = [source for source in doc.spans if len(doc.spans[source]) > 0
                       and not doc.spans[source].attrs.get("aggregated", False)
                       and not doc.spans[source].attrs.get("avoid_in_aggregation", False)]

            if len(sources) > 0:
                df = model.get_observation_df(doc)
                # Running the actual aggregation
                agg_df = model._aggregate(df)
                # Converting back to token labels
                preds.append(agg_df.values.argmax(axis=1).tolist())
            else:
                preds.append([0] * len(doc))

        return preds
