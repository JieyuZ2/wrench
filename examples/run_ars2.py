import logging
import argparse
import torch
import json
import sys

from wrench.dataset.utils import create_unbalanced_set
from wrench.labelmodel import Snorkel

# BERT: bert-base, roberta-base
default_parameter = {
    'batch_size'               : 16,
    'real_batch_size'          : 16,
    'test_batch_size'          : 256,
    'n_steps'                  : 10000,
    'grad_norm'                : -1,
    'use_lr_scheduler'         : True,
    'binary_mode'              : False,

    'lr_scheduler'             : 'default',
    'optimizer'                : 'default',

    'backbone'                 : 'BERT',
    'backbone_model_name'      : 'roberta-base',
    'backbone_max_tokens'      : 512,
    'backbone_fine_tune_layers': 4
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_root', type=str)
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--lm_pred_path', type=str, default='./', help="path of the prediction of label model.")
    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--optimal_params_path', type=str)
    parser.add_argument('--loss_type', type=str)
    parser.add_argument('--ir', type=str, help='imbalance ratio', required=True)
    parser.add_argument('--ars2', type=bool, default=False, help='selection for opening ars2')
    parser.add_argument('--gpu', type=int, default=0, help='id of gpu (e.g. 0)')
    args = parser.parse_args()

    # path to root
    if args.path_to_root:
        ptr = args.path_to_root
    else:
        ptr = './'

    from wrench.evaluation import AverageMeter, METRIC
    from wrench.dataset import load_dataset
    from wrench.logging import LoggingHandler

    # loss modified
    from wrench.endmodel.ars2 import ARS2

    #### Just some code to print debug information to stdout
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])

    logger = logging.getLogger(__name__)

    sys.path.append(args.path_to_root)
    dataset_path = f'{args.path_to_root}/{args.dataset_path}'
    lm_path = f'{args.path_to_root}/{args.lm_pred_path}'
    optimal_param_path = f'{args.path_to_root}/{args.optimal_params_path}'
    data = args.data
    imbalance_ratio = args.ir
    loss_type = args.loss_type
    ars2 = args.ars2

    device = torch.device(f'cuda:{args.gpu}')

    cr_rr = [(True, 'class_top'), (False, 'class_top'), (True, None)]  # ARS2, w/o RR, w/o CR

    for para in cr_rr:
        ranking_parameter = {
            'score_type': 'pred',  # pred: P(A), margin(AUM): P(A)-P(B)
            'mean_score_type': 'mean',
            're_sample_type': para[1],  # switch of class-wise ranking
            're_correction': para[0],  # switch of rule-aware ranking
            're_sample_concat': True,
            'dev_mode': False,
            'linear_ratio': 1,
            'score_threshold': 0,
        }

        #### Load dataset
        m_name = '_roberta'
        train_data, valid_data, test_data = load_dataset(dataset_path, data, extract_feature=False)

        #### Run label model: Snorkel
        label_model = Snorkel(
            lr=0.01,
            l2=0.0,
            n_epochs=10
        )
        label_model.fit(
            dataset_train=train_data,
            dataset_valid=valid_data
        )
        acc = label_model.test(test_data, 'acc')

        #### Create imbalanced dataset
        imbalance_ids_train = create_unbalanced_set(train_data, imbalance_ratio)
        imbalance_ids_valid = create_unbalanced_set(valid_data, imbalance_ratio)
        train_data = train_data.create_subset(imbalance_ids_train).get_covered_subset()
        valid_data = valid_data.create_subset(imbalance_ids_valid)
        aggregated_hard_labels = label_model.predict(train_data)

        #### Load optimized params
        with open(f"{optimal_param_path}/optimized{m_name}_res_imbalance{imbalance_ratio}_{data}.json", "r") as load_f:
            optimized_param = json.load(load_f)

        #### Run end model: RoBERTa
        print(optimized_param[loss_type])
        log = {}
        meter = AverageMeter(names=METRIC.keys())
        for i in range(5):
            model = ARS2(
                **default_parameter,
                **ranking_parameter,
                **optimized_param[loss_type]
            )
            history = model.fit(
                dataset_train=train_data,
                dataset_name=data,
                y_train=aggregated_hard_labels,
                dataset_valid=valid_data,
                evaluation_step=5,

                distillation=ars2,
                score_step=5,
                avg_score_step=10,

                metric='f1_macro',
                patience=40,
                device=device
            )
            proba_y = model.predict_proba(test_data)
            metrics = {metric: metric_fn(test_data.labels, proba_y) for metric, metric_fn in METRIC.items()}
            logger.info(metrics)
            log[i] = {
                'metrics': metrics,
                'history': history,
            }
            meter.update(**metrics)

        metrics = meter.get_results()
        logger.info(metrics)
        best_log = {
            'metrics': metrics,
            'exp': log,
            'paras': optimized_param[loss_type]}

        cr_rr_opt = '_'.join([str(x) for x in para])
        json.dump(best_log, open(f'{args.path_to_root}/eval_res/inference_distillation_BERT_'
                                 f'{data}_{imbalance_ratio}_{loss_type}_{cr_rr_opt}.json', 'w'), indent=4)
