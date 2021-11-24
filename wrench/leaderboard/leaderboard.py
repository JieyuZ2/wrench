# import logging
import torch
from ..dataset import load_dataset
# from wrench.classification import WeaSEL
# from wrench.labelmodel import Snorkel
# from wrench.endmodel import EndClassifierModel

class ModelWrapper():
    """Model wrapper such that we can compare 2stage and end2end models.
    These"""
    def __init__(self, model_func, name, label_model_func=None):
        self.model_func=model_func
        self.label_model_func=label_model_func
        self.name = name
        self.reset()
    
    def reset(self):
        self.model=self.model_func()
        if self.label_model_func is not None:
            self.label_model=self.label_model_func()
        else:
            self.label_model=None

    def fit(self, train_data, valid_data, metric, evaluation_step=10, patience=100, device='cpu'):
        if self.label_model is not None:
            # 2stage model
            self.label_model.fit(
                dataset_train=train_data,
                dataset_valid=valid_data
                )
            train_data = train_data.get_covered_subset()
            soft_labels = self.label_model.predict_proba(train_data)
            self.model.fit(
                dataset_train=train_data,
                y_train=soft_labels,
                dataset_valid=valid_data,
                evaluation_step=evaluation_step,
                metric=metric,
                patience=patience,
                device=device
            )
        else:
          self.model.fit(
                dataset_train=train_data,
                dataset_valid=valid_data,
                evaluation_step=evaluation_step,
                metric=metric,
                patience=patience,
                device=device
            )

    def test(self, metrics, test_data):
        metrics = self.model.test(test_data, metrics)
        return metrics
    
    def save(self, save_dir, dataset_name):
        name = self.name + '_' + dataset_name
        if self.label_model is not None:
            self.label_model.save(save_dir + name + '.label')
        torch.save(self.model.model.state_dict(), save_dir + name + '.pt')



def make_leaderboard(models, datasets, metrics, dataset_path='../data/', device='cpu', save_dir=None):
    """Trains each model on each datasets and evaluates the different metrics."""
    results = []
    for dataset in datasets:
        train_data, valid_data, test_data = load_dataset(
            dataset_path,
            dataset,
            extract_feature=True,
            extract_fn='bert',
            cache_name='bert'
        )
        dataset_results = []
        for model in models:
            model.fit(train_data, valid_data, metrics[0], device=device)
            dataset_results.append(model.test(metrics, test_data))
            if save_dir is not None:
                model.save(save_dir, dataset)
        results.append(dataset_results)
    return results


# if __name__=="__main__":
#     device='cpu'
#     datasets = ['youtube', 'sms']
#     model1 = ModelWrapper(
#         model_func=lambda : WeaSEL(
#             temperature=1.0,
#             dropout=0.3,
#             hidden_size=100,

#             batch_size=16,
#             real_batch_size=8,
#             test_batch_size=128,
#             n_steps=1000,
#             grad_norm=1.0,

#             backbone='MLP',
#             # backbone='BERT',
#             backbone_model_name='MLP',
#             backbone_fine_tune_layers=-1,  # fine  tune all
#             optimizer='AdamW',
#             optimizer_lr=5e-5,
#             optimizer_weight_decay=0.0,
#         )
#     )
#     model2 = ModelWrapper(
#         model_func=lambda : EndClassifierModel(
#             batch_size=128,
#             test_batch_size=512,
#             n_steps=1000,
#             backbone='MLP',
#             optimizer='Adam',
#             optimizer_lr=1e-2,
#             optimizer_weight_decay=0.0,
#         ),
#         label_model_func=lambda: Snorkel(
#             lr=0.01,
#             l2=0.0,
#             n_epochs=10
#         )
#     )
#     models = [model1, model2]
#     metrics = ['acc']
#     print(leaderboard(models=models, datasets=datasets, metrics=metrics, dataset_path='../../datasets/'))
