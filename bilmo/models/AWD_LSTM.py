from bilmo.scripts.config import Config
from fastai.text.models import AWD_LSTM
from torch import nn
import logging
conf = Config.conf
log = logging.getLogger("cafa-logger")


def get_AWD_LSTM_config():
    return conf['AWD_STM_config']


my_AWD_LSTM = AWD_LSTM
