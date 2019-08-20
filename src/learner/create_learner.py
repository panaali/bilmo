from src.scripts.config import Config
from src.models.my_AWD_LSTM import get_AWD_LSTM_config, AWD_LSTM
from src.models.my_Transformer import get_transformerXL_config, get_transformer_config, Transformer, TransformerXL
from src.callbacks.killer import KillerCallback
from src.optimizer.radam import RAdam
from fastai.callback import AdamW
from src.metrics.f1 import f1
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
    if conf['network'] == 'my_AWD_LSTM':
        network_config = get_AWD_LSTM_config()
    else:
        raise BaseException('network ' + conf['network'] + 'not defined')

    learn_cls = text_classifier_learner(
        data_cls, eval(conf['network'].replace('my_','')), config=network_config, drop_mult=0.1, pretrained=False, opt_func=get_optimizer(),
        metrics=[accuracy, f1], callback_fns=[partial(CSVLogger, append=True), KillerCallback])
    
    if conf['fp16']:
        learn_cls.to_fp16()
    
    if conf['lm_encoder_path'] is not None and conf['vocab_path'] is not None:
        learn_cls.load_encoder(conf['lm_encoder_path'])

    return learn_cls
