from wrench.classification import WeaSEL
from wrench.labelmodel import Snorkel
from wrench.endmodel import EndClassifierModel
from wrench.leaderboard import ModelWrapper, make_leaderboard

if __name__=="__main__":
    device='cpu'
    datasets = ['youtube', 'sms']
    model1 = ModelWrapper(
        model_func=lambda : WeaSEL(
            temperature=1.0,
            dropout=0.3,
            hidden_size=100,

            batch_size=16,
            real_batch_size=8,
            test_batch_size=128,
            n_steps=1000,
            grad_norm=1.0,

            backbone='MLP',
            # backbone='BERT',
            backbone_model_name='MLP',
            backbone_fine_tune_layers=-1,  # fine  tune all
            optimizer='AdamW',
            optimizer_lr=5e-5,
            optimizer_weight_decay=0.0,
        ),
        name="MLP_weasel"
    )
    model2 = ModelWrapper(
        model_func=lambda : EndClassifierModel(
            batch_size=128,
            test_batch_size=512,
            n_steps=1000,
            backbone='MLP',
            optimizer='Adam',
            optimizer_lr=1e-2,
            optimizer_weight_decay=0.0,
        ),
        label_model_func=lambda: Snorkel(
            lr=0.01,
            l2=0.0,
            n_epochs=10
        ),
        name = '2stage_MLP_snorkel'
    )
    models = [model1, model2]
    metrics = ['acc']
    print(make_leaderboard(models=models, datasets=datasets, metrics=metrics, dataset_path='../../datasets/',
        save_dir='../../saved_models/'))