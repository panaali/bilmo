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
from datetime import datetime
import pickle
import numpy as np
import os
from torch import nn

data_path = './data/'

class dna_tokenizer(BaseTokenizer):
    def tokenizer(self, t):
        tokens = t.split(' ')
        bos = tokens[0]
        seq = tokens[1]
        after_seq = tokens[2:-1]
        eos = tokens[-1]
        result = [bos]
        result += list(seq)  # sequence string to list
        result += after_seq
        result.append(eos)
        return result

@call_parse
def main(gpu: Param("GPU to run on", str)=None,
        max_cpu_per_dataloader: Param("Max CPU", int, opt=True)=8,
        bs: Param("batch size", int)=256,
        fp16: Param("mixed precision", int, opt=True)=0,
        sp_processor: Param("use sentence piece as processor", int)=0,
        sp_model: Param("sentence piece trained model file", str)=None,
        sp_vocab: Param("sentence piece trained model file", str)=None,
    ):
    datetime_str = f'{datetime.now():%Y-%m-%d_%H-%M-%S%z}'
    random_seed = 0
    max_vocab = 30000
    print('max_cpu_per_dataloader', max_cpu_per_dataloader, 'bs', bs,
        'fp16', fp16, 'sp_processor', sp_processor, 'sp_model', sp_model, 'sp_vocab', sp_vocab)
    """## Prepare Dataset"""
    local_project_path = './data/sprot_lm/'

    #### Distributed
    print('gpu', gpu)
    gpu = setup_distrib(gpu)
    n_gpus = num_distrib()
    if n_gpus > 0:
        workers = min(max_cpu_per_dataloader, num_cpus()//n_gpus)
    else:
        workers = min(max_cpu_per_dataloader, num_cpus())
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
                                include_eos=True), NumericalizeProcessor(max_vocab=max_vocab)]
    df = pickle.load(
        open('./data/sprot_lm/sproat_sequence_taxon_anc.pickle', 'rb'))

    if sp_processor: # './data/sprot_lm/tmp/spm.model', './data/sprot_lm/tmp/spm.vocab'
        processor = [OpenFileProcessor(), SPProcessor(sp_model=sp_model, sp_vocab=sp_vocab, max_sentence_len=35826, max_vocab_sz=max_vocab)]
    data_lm = (TextList.from_df(df, path=local_project_path, cols='seq_anc_tax', processor=processor)
                    .split_by_rand_pct(0.1, seed = random_seed)
                    .label_for_lm()
                    .databunch(bs=bs, num_workers=workers))

    data_lm.vocab.save(local_project_path +
                       'vocab_lm_sproat_seq_anc_tax-' + datetime_str + '.pickle')

    print('data_cls Training set size', len(data_lm.train_ds))
    print('data_cls Validation set size', len(data_lm.valid_ds))
    print('vocab size ', len(data_lm.vocab.itos))


    learn_lm = language_model_learner(
        data_lm, AWD_LSTM, drop_mult=0.1, pretrained=False)

    if gpu is None:
        print(gpu, 'DataParallel')
        learn_lm.model = nn.DataParallel(learn_lm.model)
    else:
        print(gpu, 'to_distributed')
        learn_lm.to_distributed(gpu)
        if fp16:
            learn_lm.to_fp16()
    

    lr = 3e-3
    print(gpu, 'freeze')
    learn_lm.freeze()
    learn_lm.fit_one_cycle(1, lr, moms=(0.8, 0.7))  # I don't know why multigpu doesn't work without first freezing

    print(gpu, 'unfreeze')
    learn_lm.unfreeze()
    learn_lm.fit_one_cycle(10, lr*10, moms=(0.8, 0.7))
    learn_lm.save('lm-sp-anc-v1-1-' + datetime_str)
    learn_lm.save_encoder('lm-sp-ans-v1-1-enc-' + datetime_str)
    
    learn_lm.fit_one_cycle(10, lr, moms=(0.8, 0.7))
    learn_lm.save('lm-sp-anc-v1-2-' + datetime_str)
    learn_lm.save_encoder('lm-sp-ans-v1-2-enc-' + datetime_str)

    learn_lm.fit_one_cycle(10, lr/10, moms=(0.8, 0.7))
    learn_lm.save('lm-sp-anc-v1-3' + datetime_str)
    learn_lm.save_encoder('lm-sp-ans-v1-3-enc-' + datetime_str)
    learn_lm.export(file = 'export-lm-sp-ans-v1-3' + datetime_str+ '.pkl')
    print('Done')

# main(None)
