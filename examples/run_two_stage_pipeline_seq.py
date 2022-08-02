import logging
import torch
from wrench.dataset.utils import get_glove_embedding
from wrench.dataset import load_dataset
from wrench._logging import LoggingHandler
from wrench.seq_labelmodel import HMM
from wrench.seq_endmodel import LSTMTaggerModel, BERTTaggerModel

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

logger = logging.getLogger(__name__)

device = torch.device('cuda')

#### Load dataset
dataset_path = '../datasets/'
data = 'laptopreview'
train_data, valid_data, test_data = load_dataset(dataset_path, data, extract_feature=False)

#### Run label model: HMM
label_model = HMM(
    n_epochs=10
)
label_model.fit(
    dataset_train=train_data,
    dataset_valid=valid_data
)
f1 = label_model.test(test_data, 'f1_seq')
logger.info(f'label model test f1: {f1}')

#### Get training labels
aggregated_labels = label_model.predict(train_data)

#### Run end model: LSTM
word_dict, embedding = get_glove_embedding(
    embedding_file_path='../datasets/glove.6B.100d.txt',
    PAD=train_data.PAD,
    UNK=train_data.UNK
)
train_data.load_embed_dict(word_embed_dict=word_dict)
valid_data.load_embed_dict(word_embed_dict=word_dict)
test_data.load_embed_dict(word_embed_dict=word_dict)
model = LSTMTaggerModel(
    batch_size=32,
    test_batch_size=512,
    n_steps=10000,

    lr=1e-2,
    l2=1e-8,
    word_emb_dim=100,
)
model.fit(
    dataset_train=train_data,
    y_train=aggregated_labels,
    dataset_valid=valid_data,
    evaluation_step=10,
    metric='f1_seq',
    patience=100,
    device=device
)
f1 = model.test(test_data, 'f1_seq')
logger.info(f'end model (LSTM) test f1: {f1}')

#### Run end model: BERT
model = BERTTaggerModel(
    batch_size=32,
    test_batch_size=512,
    n_steps=10000,

    lr=2e-5,
    l2=1e-6,
    use_crf=True
)
model.fit(
    dataset_train=train_data,
    y_train=aggregated_labels,
    dataset_valid=valid_data,
    evaluation_step=10,
    metric='f1_seq',
    patience=100,
    device=device
)
f1 = model.test(test_data, 'f1_seq')
logger.info(f'end model (BERT) test f1: {f1}')

