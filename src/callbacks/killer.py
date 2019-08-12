import os
import torch
from fastai.basic_train import LearnerCallback
from fastai.torch_core import *

__all__ = ['KillerCallback']

class KillerCallback(LearnerCallback):
    def check_for_killme(self, **kwargs: Any) -> None:
        if os.path.isfile('kill.me'):
            num_gpus = torch.cuda.device_count()
            for gpu_id in range(num_gpus):
                try:
                    torch.cuda.set_device(gpu_id)
                    torch.cuda.empty_cache()
                except:
                    pass
            exit(0)

    def on_epoch_end(self, **kwargs: Any) -> None:
        self.check_for_killme()
