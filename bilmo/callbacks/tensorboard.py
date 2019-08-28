from bilmo.scripts.config import Config
from fastai.callbacks.tensorboard import *
from tensorboardX import SummaryWriter
# from torch.utils.tensorboard import SummaryWriter #Alternative SummaryWriter from official pytorch
from functools import partial
from pathlib import Path
conf = Config.conf

def add_tensorboard(learn):
    tensorboard_logs_path = Path('./tensorboard-logs/' +
                                 conf['project_name'])

    def on_train_begin(self, **kwargs):
        """disables graph writing"""
        pass

    if not conf['log_graph_tensorboard']:
        LearnerTensorboardWriter.on_train_begin = on_train_begin
    learn.callback_fns.append(
        partial(LearnerTensorboardWriter,
                base_dir=tensorboard_logs_path,
                name=conf['datetime']))
