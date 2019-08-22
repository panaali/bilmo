from bilmo.scripts.config import Config
from bilmo.models.AWD_LSTM import get_AWD_LSTM_config, my_AWD_LSTM
from bilmo.models.Transformer import get_transformerXL_config, get_transformer_config, my_Transformer, my_TransformerXL
from bilmo.callbacks.killer import KillerCallback
from bilmo.optimizer.radam import RAdam
from fastai.callback import AdamW
from bilmo.metrics.f1 import f1
from fastai.callbacks.csv_logger import CSVLogger
from fastai.text.learner import text_classifier_learner
from fastai.metrics import accuracy
from functools import partial
from torch import nn
import logging

conf = Config.conf
log = logging.getLogger("cafa-logger")

def get_optimizer():
    if conf['optimizer'] == 'radam':
        return partial(RAdam)
    else:
        return partial(AdamW)

def create_learner(data_cls):
    if conf['network'] == 'my_Transformer':
        network_config = get_transformer_config()
    elif conf['network'] == 'my_TransformerXL':
        network_config = get_transformerXL_config()
    elif conf['network'] == 'my_AWD_LSTM':
        network_config = get_AWD_LSTM_config()
    else:
        raise BaseException('network ' + conf['network'] + ' not defined')

    learn_cls = text_classifier_learner(
        data_cls, eval(conf['network']), config=network_config, drop_mult=0.1, pretrained=False, opt_func=get_optimizer(),
        metrics=[accuracy, f1], callback_fns=[partial(CSVLogger, append=True), KillerCallback])
    
    if conf['fp16']:
        learn_cls.to_fp16()
    
    if conf['lm_encoder_path'] is not None and conf['vocab_path'] is not None:
        learn_cls.load_encoder(conf['lm_encoder_path'])

    return learn_cls
