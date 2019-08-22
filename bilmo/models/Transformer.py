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


def get_transformer_config():
    return dict(ctx_len=512, n_layers=12, n_heads=12, d_model=768, d_head=64, d_inner=3072, resid_p=0.1, attn_p=0.1,
                            ff_p=0.1, embed_p=0.1, output_p=0., bias=True, scale=True, act=Activation.GeLU, double_drop=False,
                            init=init_transformer, mask=False)
def get_transformerXL_config():
    return dict(ctx_len=150, n_layers=12, n_heads=10, d_model=410, d_head=41, d_inner=2100, resid_p=0.1, attn_p=0.1,
                            ff_p=0.1, embed_p=0.1, output_p=0.1, bias=False, scale=True, act=Activation.ReLU, double_drop=True,
                            init=init_transformer, mem_len=150, mask=False)


my_Transformer = Transformer
my_TransformerXL = TransformerXL
