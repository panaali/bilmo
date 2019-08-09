import os
import torch
__all__ = ['KillerCallback']

class KillerCallback(Callback):
    def check_for_killme():
        if os.path.isfile('kill.me'):
            num_gpus = torch.cuda.device_count()
            for gpu_id in range(num_gpus):
                try:
                    torch.cuda.set_device(gpu_id)
                    torch.cuda.empty_cache()
                except:
                    pass
            exit(0)

    def on_batch_begin(self, **kwargs: Any) -> None:
        check_for_killme()

    def on_batch_end(self, **kwargs: Any) -> None:
        check_for_killme()
