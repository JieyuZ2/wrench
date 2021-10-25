import json
from typing import Any, Optional

DEFAULT_OPTIMIZER_CONFIG = {
    'name' : 'Adam',
    'paras': {
        'lr'          : 1e-3,
        'weight_decay': 0.0,
    }
}

DEFAULT_LR_SCHEDULER_CONFIG = {
    'name' : 'StepLR',
    'paras': {
        'step_size': 1000,
    }
}

DEFAULT_LOGREG_CONFIG = {

}

DEFAULT_MLP_CONFIG = {
    'hidden_size': 100,
    'dropout'    : 0.0,
}

DEFAULT_IMAGE_CLASSIFIER_CONFIG = {
    'model_name': 'resnet18',
}

DEFAULT_BERT_CONFIG = {
    'model_name'      : 'bert-base-cased',
    'max_tokens'      : 512,
    'fine_tune_layers': -1,
}

DEFAULT_BACKBONE_CONFIG = {
    'name' : 'MLP',
    'paras': DEFAULT_MLP_CONFIG
}

DEFAULT_BACKBONE_MAP = {
    'MLP'            : DEFAULT_MLP_CONFIG,
    'LogReg'         : DEFAULT_LOGREG_CONFIG,
    'ImageClassifier': DEFAULT_IMAGE_CLASSIFIER_CONFIG,
    'BERT'           : DEFAULT_BERT_CONFIG,
}

DEFAULT_LABEL_MODEL_CONFIG = {
    'name' : 'MajorityVoting',
    'paras': {}
}


class Config:
    def __init__(self,
                 hyperparas: dict,
                 prefix: Optional[str] = '',
                 use_optimizer: Optional[bool] = False,
                 use_lr_scheduler: Optional[bool] = False,
                 use_backbone: Optional[bool] = False,
                 use_label_model: Optional[bool] = False,
                 **kwargs: Any
                 ):

        self.hyperparas = hyperparas
        self.prefix = prefix

        if use_optimizer:
            self.optimizer_config = DEFAULT_OPTIMIZER_CONFIG.copy()

            if use_lr_scheduler:
                self.lr_scheduler_config = DEFAULT_LR_SCHEDULER_CONFIG.copy()

        if use_backbone:
            self.backbone_config = DEFAULT_BACKBONE_CONFIG.copy()

        if use_label_model:
            self.label_model_config = DEFAULT_LABEL_MODEL_CONFIG.copy()

        self.update(**kwargs)

    def update(self, **kwargs):
        prefix = self.prefix
        if prefix != '':
            prefix += '_'
            kwargs = {k.replace(prefix, ''): v for k, v in kwargs.items() if k.startswith(prefix)}

        if hasattr(self, 'optimizer_config'):
            if 'optimizer' in kwargs:
                optimizer_ = kwargs['optimizer']
                if optimizer_ != self.optimizer_config['name']:
                    self.optimizer_config['name'] = optimizer_
                    self.optimizer_config['paras'] = {}

            for k, v in kwargs.items():
                if k.startswith('optimizer_'):
                    k = k.replace('optimizer_', '')
                    self.optimizer_config['paras'][k] = v

            if hasattr(self, 'lr_scheduler_config'):
                if 'lr_scheduler' in kwargs:
                    lr_scheduler_ = kwargs['lr_scheduler']
                    if lr_scheduler_ != self.lr_scheduler_config['name']:
                        self.lr_scheduler_config['name'] = lr_scheduler_
                        self.lr_scheduler_config['paras'] = {}

                for k, v in kwargs.items():
                    if k.startswith('lr_scheduler_'):
                        k = k.replace('lr_scheduler_', '')
                        self.lr_scheduler_config['paras'][k] = v

        if hasattr(self, 'backbone_config'):
            if 'backbone' in kwargs:
                backbone_ = kwargs['backbone']
                if backbone_ != self.backbone_config['name']:
                    self.backbone_config['name'] = backbone_
                    self.backbone_config['paras'] = DEFAULT_BACKBONE_MAP[kwargs['backbone']].copy()

            for k, v in kwargs.items():
                if k.startswith('backbone_'):
                    k = k.replace('backbone_', '')
                    self.backbone_config['paras'][k] = v

        if hasattr(self, 'label_model_config'):
            if 'label_model' in kwargs:
                label_model_ = kwargs['label_model']
                if label_model_ != self.label_model_config['name']:
                    self.label_model_config['name'] = label_model_
                    self.label_model_config['paras'] = {}

            for k, v in kwargs.items():
                if k.startswith('label_model_'):
                    k = k.replace('label_model_', '')
                    self.label_model_config['paras'][k] = v

        for k, v in self.hyperparas.items():
            if k in kwargs: self.hyperparas[k] = kwargs[k]

        return self

    def __repr__(self):
        s = '\n'
        if self.prefix != '':
            prefix = self.prefix + ': '
        else:
            prefix = ''

        s += '=' * 10 + f'[{prefix}hyper parameters]' + '=' * 10 + '\n'
        s += json.dumps(self.hyperparas, indent=4) + '\n'

        if hasattr(self, 'optimizer_config'):
            s += '=' * 10 + f'[{prefix}optimizer config]' + '=' * 10 + '\n'
            s += json.dumps(self.optimizer_config, indent=4) + '\n'

        if hasattr(self, 'lr_scheduler_config'):
            s += '=' * 10 + f'[{prefix}lr scheduler config]' + '=' * 10 + '\n'
            s += json.dumps(self.lr_scheduler_config, indent=4) + '\n'

        if hasattr(self, 'backbone_config'):
            s += '=' * 10 + f'[{prefix}backbone config]' + '=' * 10 + '\n'
            s += json.dumps(self.backbone_config, indent=4) + '\n'

        if hasattr(self, 'label_model_config'):
            s += '=' * 10 + f'[{prefix}label model_config config]' + '=' * 10 + '\n'
            s += json.dumps(self.label_model_config, indent=4) + '\n'

        return s
