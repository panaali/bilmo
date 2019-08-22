from bilmo.scripts.config import Config
from fastai.text.models import AWD_LSTM
from torch import nn
import logging
conf = Config.conf
log = logging.getLogger("cafa-logger")


def get_AWD_LSTM_config():
    return dict(emb_sz=400, n_hid=1152, n_layers=3, pad_token=1, qrnn=False,
                        bidir=False, output_p=0.4, hidden_p=0.3, input_p=0.4, embed_p=0.05, weight_p=0.5)


my_AWD_LSTM = AWD_LSTM
