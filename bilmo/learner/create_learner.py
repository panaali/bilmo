from bilmo.scripts.config import Config
conf = Config.conf
if conf['comet.ml']:
    from comet_ml import Experiment

from bilmo.models.AWD_LSTM import get_AWD_LSTM_config, my_AWD_LSTM
from bilmo.models.Transformer import get_transformerXL_config, get_transformer_config, my_Transformer, my_TransformerXL
from bilmo.callbacks.killer import KillerCallback
from bilmo.optimizer.radam import RAdam
from fastai.callback import AdamW
from fastai.callbacks import OverSamplingCallback
from bilmo.metrics.MultiLabelFbeta import addf1MultiLabel
from fastai.callbacks.csv_logger import CSVLogger
from fastai.layers import BCEWithLogitsFlat
from fastai.text.learner import text_classifier_learner
from fastai.metrics import accuracy, FBeta

from functools import partial
from torch import nn
import torch
import numpy as np
import logging



log = logging.getLogger("cafa-logger")

def get_optimizer():
    if conf['optimizer'] == 'radam':
        return partial(RAdam)
    else:
        return partial(AdamW)

def get_metrics():
    if conf['classificiation_type'] == 'binary':
        f1 = FBeta(average='macro', beta=1)
        return [accuracy, f1]

def append_callback_fns(learn_cls):
    if conf['OverSamplingCallback'] and conf['classificiation_type'] != 'multilabel':
        learn_cls.callbacks(OverSamplingCallback(learn_cls, weights=None))
    if conf['classificiation_type'] == 'multilabel':
        addf1MultiLabel(learn_cls)


def get_weights(data_cls):
    labels = data_cls.train_dl.dataset.y.items
    labels_flatten= [k for j in labels for k in j]
    _, counts = np.unique(labels_flatten, return_counts=True)
    weights = torch.FloatTensor((1 / counts))
    weights = weights.expand(conf['bs'], -1).flatten().cuda() # bs is different for validation, this should be calculated inside the callback not here
    return weights

def get_callback_fns():
    return [
        partial(CSVLogger,
                append=True,
                filename='../../' + conf['log_path'] + conf['log_filename']),
        KillerCallback
    ]

def get_loss_func():
    if conf['loss_func'] == 'BCEWithLogitsFlat':
        return BCEWithLogitsFlat(weight=None, reduction=conf['loss_reduction'])
    elif conf['loss_func'] == 'MultiLabelCrossEntropy':
        # https://discuss.pytorch.org/t/how-should-i-implement-cross-entropy-loss-with-continuous-target-outputs/10720/17
        def cross_entropy(pred, targ):
            softmax = nn.Softmax(dim=1)
            soft_targets = softmax(targ)
            logsoftmax = nn.LogSoftmax(dim=1)
            return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))
        return cross_entropy



def create_learner(data_cls):
    if conf['network'] == 'my_Transformer':
        network_config = get_transformer_config()
    elif conf['network'] == 'my_TransformerXL':
        network_config = get_transformerXL_config()
    elif conf['network'] == 'my_AWD_LSTM':
        network_config = get_AWD_LSTM_config()
    else:
        raise BaseException('network ' + conf['network'] + ' not defined')

    if conf['comet.ml']:
        experiment = Experiment(api_key="MjxiBuEhQzcaS2JivObmZjrP9",
                                project_name="bilmo",
                                workspace="panaali")

    weights = get_weights(data_cls)
    learn_cls = text_classifier_learner(data_cls,
                                        eval(conf['network']),
                                        config=network_config,
                                        drop_mult=conf['drop_mult'],
                                        loss_func=get_loss_func(),
                                        pretrained=False,
                                        opt_func=get_optimizer(),
                                        metrics=get_metrics(),
                                        wd=conf['weight_decay'],
                                        callback_fns=get_callback_fns())
    append_callback_fns(learn_cls)
    if conf['fp16']:
        learn_cls.to_fp16()

    if conf['lm_encoder_path'] is not None and conf['vocab_path'] is not None:
        learn_cls.load_encoder(conf['lm_encoder_path'])

    return learn_cls
