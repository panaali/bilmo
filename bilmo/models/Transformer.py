from bilmo.scripts.config import Config
from fastai.text.models import TransformerXL, Transformer, Activation
from torch import nn
import logging
conf = Config.conf
log = logging.getLogger("cafa-logger")

def init_transformer(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.normal_(m.weight, 0., 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias, 0.)
    elif classname.find('LayerNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.normal_(m.weight, 1., 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias, 0.)
    elif classname.find('TransformerXL') != -1:
        if hasattr(m, 'u'):
            nn.init.normal_(m.u, 0., 0.02)
        if hasattr(m, 'v'):
            nn.init.normal_(m.v, 0., 0.02)


"""
https://docs.fast.ai/text.models.html
"""
def get_transformer_config():
    T_config = conf['Transformer_config']
    T_config['init'] = eval(T_config['init'])
    T_config['act'] = eval(T_config['act'])
    return T_config

def get_transformerXL_config():
    TXL_config = conf['TransformerXL_config']
    TXL_config['init'] = eval(TXL_config['init'])
    TXL_config['act'] = eval(TXL_config['act'])
    return TXL_config

my_Transformer = Transformer
my_TransformerXL = TransformerXL
