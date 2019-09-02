from bilmo.scripts.config import Config
from fastai.callbacks.tensorboard import LearnerTensorboardWriter, ModelStatsTBRequest
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

    if conf['fp16']: # convert float16 gradients to float32
        def ModelStatsTBRequestInit(self, model:nn.Module, iteration:int, tbwriter:SummaryWriter, name:str):
            super().__init__(tbwriter=tbwriter, iteration=iteration)
            self.gradients = [x.grad.clone().detach().float().cpu() for x in model.parameters() if x.grad is not None]
            self.name = name
        ModelStatsTBRequest.__init__ = ModelStatsTBRequestInit

    learn.callback_fns.append(
        partial(LearnerTensorboardWriter,
                base_dir=tensorboard_logs_path,
                name=conf['datetime']))
