# from fastai.script import *
# from fastai.metrics import *
from fastai.text import *
from fastai import *
from fastai.distributed import setup_distrib
from fastai.text import Tokenizer
from fastai.text.data import TokenizeProcessor, NumericalizeProcessor
from fastai.text.data import TextList
from fastai.text.learner import language_model_learner
from fastai.text.models import AWD_LSTM
from fastai.text.transform import BaseTokenizer
from fastai.torch_core import num_distrib
from fastai.core import num_cpus
from fastai.script import call_parse, Param
import pickle
import numpy as np
import os
from torch import nn

data_path = './data/'
bs = 256
fp16 = False
sp_processor = True

class dna_tokenizer(BaseTokenizer):
    def tokenizer(self, t):
        tokens = t.split(' ')
        before_seq = tokens[:-2] #bug!! not compatible with new tax anc
        seq = tokens[-2]
        eos = tokens[-1]
        result = before_seq
        result += list(seq)  # sequence string to list
        result.append(eos)
        return result


@call_parse
def main(gpu: Param("GPU to run on", str) = None):

    random_seed = 0
    min_cpu_per_dataloader = 16
    """## Prepare Dataset"""
    local_path = './'
    local_project_path = local_path + 'data/sprot_lm/'

    #### Distributed
    print('gpu', gpu)
    gpu = setup_distrib(gpu)
    n_gpus = num_distrib()
    if n_gpus > 0:
        workers = min(min_cpu_per_dataloader, num_cpus()//n_gpus)
    else:
        workers = num_cpus()
    print(gpu, 'n_gpus', n_gpus)
    print(gpu, 'workers', workers)


    """## Prepare fastai"""
    np.random.seed(random_seed)

    if not os.path.exists(local_project_path):
        os.makedirs(local_project_path)
    print('local_project_path:', local_project_path)

    """## Tokenization"""
    tokenizer = Tokenizer(tok_func=dna_tokenizer, pre_rules=[],
                        post_rules=[], special_cases=[])
    processor = [TokenizeProcessor(tokenizer=tokenizer, include_bos=True,
                                include_eos=True), NumericalizeProcessor(max_vocab=30000)]
    df = pickle.load(
        open('./data/sprot_lm/sproat_sequence_taxon_anc.pickle', 'rb'))

    if sp_processor:
        processor = [OpenFileProcessor(), SPProcessor()]
    data_lm = (TextList.from_df(df, path=local_project_path, cols='seq_anc_tax', processor=processor)
                    .split_by_rand_pct(0.1, seed = random_seed)
                    .label_for_lm()
                    .databunch(bs=bs, num_workers=workers))

    data_lm.vocab.save(local_project_path +
                       'vocab_lm_sproat_seq_anc_tax_spprocessor.pickle')

    print('data_cls Training set size', len(data_lm.train_ds))
    print('data_cls Validation set size', len(data_lm.valid_ds))
    print('vocab size ', len(data_lm.vocab.itos))


    learn_lm = language_model_learner(
        data_lm, AWD_LSTM, drop_mult=0.1, pretrained=False)

    if gpu is None:
        # print(gpu, 'DataParallel')
        # learn_lm.model = nn.DataParallel(learn_lm.model)
        pass
    else:
        print(gpu, 'to_distributed')
        learn_lm.to_distributed(gpu)
        if fp16:
            learn_lm.to_fp16()
    

    lr = 3e-3
    print(gpu, 'freeze')
    learn_lm.freeze()
    learn_lm.fit_one_cycle(1, lr, moms=(0.8, 0.7))

    print(gpu, 'unfreeze')
    learn_lm.unfreeze()
    learn_lm.fit_one_cycle(10, lr*10, moms=(0.8, 0.7))
    learn_lm.save('lm-sp-anc-v1-1')
    learn_lm.save_encoder('lm-sp-ans-v1-1-enc')
    
    learn_lm.fit_one_cycle(10, lr, moms=(0.8, 0.7))
    learn_lm.save('lm-sp-anc-v1-2')
    learn_lm.save_encoder('lm-sp-ans-v1-2-enc')

    learn_lm.fit_one_cycle(10, lr/10, moms=(0.8, 0.7))
    learn_lm.save('lm-sp-anc-v1-3')
    learn_lm.save_encoder('lm-sp-ans-v1-3-enc')
    print('Done')

# main(None)
