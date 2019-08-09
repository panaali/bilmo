#%%
from fastai.distributed import setup_distrib
from fastai.text import Tokenizer
from fastai.text.data import TokenizeProcessor, NumericalizeProcessor, OpenFileProcessor, SPProcessor
from fastai.text.data import TextList
from fastai.text.learner import text_classifier_learner
from fastai.text.models import AWD_LSTM, TransformerXL, Transformer
from fastai.text.transform import BaseTokenizer, Vocab
from fastai.torch_core import num_distrib, np_func
from fastai.core import num_cpus
from fastai.script import call_parse, Param
from fastai.metrics import accuracy, FBeta
from fastai.callbacks.csv_logger import CSVLogger
from functools import partial
from datetime import datetime
import pickle
import numpy as np
import os
from torch import nn
import pandas as pd
import inspect
from callbacks import *

"""
Classifier
"""
#%%
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
#%%
@call_parse
def main(train_df_path: Param("location of the training dataframe", str, opt=False),
        gpu: Param("GPU to run on", str) = None,
        max_cpu_per_dataloader: Param("Max CPU", int)=8,
        bs: Param("batch size", int)=256,
        fp16: Param("mixed precision", int)=0,
        use_sp_processor: Param("use sentence piece as processor", int)=0,
        sp_model: Param("sentence piece trained model file", str)=None,
        sp_vocab: Param("sentence piece trained vocab file", str)=None,
        lm_encoder: Param("language modeling encoder file", str)=None,
        sequence_col_name: Param("name of the sequence column",
                        str) = 'seq_anc_tax',
        label_col_name: Param('label column name', str) = 'selected_go',
        selected_go: Param('which Go_id for binary classfication. selected_go vs others', str)=None,
        selected_go2: Param('which Go_id for binary classfication. selected_go vs selected_go2', str)=None,
        vocab: Param('vocab file', str) = None,
        benchmarking: Param('benchmarking', int) = 0,
        network: Param('Which network to use? AWD_LSTM, Transformer, TransformerXL', str) = 'AWD_LSTM'
    ):
#%%
    
    # For iPython testing only
    # data_path = '../cafa/data/'
    # max_cpu_per_dataloader = 8
    # bs = 256
    # fp16 = 0
    # use_sp_processor = 0
    # sp_model = None
    # sp_vocab = None
    # gpu = None
    # train_df_path = data_path + \
    #     'cafa3/CAFA 3 Protein Targets/CAFA3_training_data/cafa_train_enhanced.p'
    # sequence_col_name = 'seq_anc_tax'
    # vocab = data_path + 'sprot_lm/vocab_lm_sproat_seq_anc_tax.pickle'
    # label_col_name = 'selected_go'
    # selected_go = 'GO:0017076'
    # lm_encoder = 'lm-sp-ans-v1-5-enc'

#%%
    datetime_str = f'{datetime.now():%Y-%m-%d_%H-%M-%S%z}'
    random_seed = 42
    max_vocab = 30000
#%%
    if gpu == '0':
        frame = inspect.currentframe()
        args, _, _, values = inspect.getargvalues(frame)
        print('function name "%s"' % inspect.getframeinfo(frame)[2])
        for i in args:
            print("    %s = %s" % (i, values[i]))
    """## Prepare Dataset"""
    local_project_path = data_path + 'sprot_lm/'
#%%
    #### Distributed
    print('gpu', gpu)
    n_gpus = num_distrib()
    gpu = setup_distrib(gpu)
    if n_gpus < 2:
        os.environ["WORLD_SIZE"] = '0'
    if n_gpus > 0:
        workers = min(max_cpu_per_dataloader, num_cpus()//n_gpus)
    else:
        workers = min(max_cpu_per_dataloader, num_cpus())
    print(gpu, 'n_gpus', n_gpus)
    print(gpu, 'workers', workers)

#%%
    """## Prepare path"""
    np.random.seed(random_seed)

    if not os.path.exists(local_project_path):
        os.makedirs(local_project_path)
    print('local_project_path:', local_project_path)

    
#%%
    df = pickle.load(open(train_df_path, 'rb'))
    print(df.columns)
    print('total number of rows', len(df))
#%%
    df = df.dropna(subset=['seq_anc_tax'])
    print('total number of rows after removing Nan', len(df))
    if label_col_name == 'selected_go':
        def find_go(row):
            if selected_go in row.go:
                res = 'T'
            else:
                res = 'F'
            return res
        df['selected_go'] = df.apply(find_go, axis=1)
        available_T = (df['selected_go'] == 'T').sum()
        if gpu == 0:
            print('number of rows that has', selected_go, 'is: ', available_T)
        df_undersampled_F = df[df['selected_go'] == 'F'][:available_T].copy()
        df_undersampled_T = df[df['selected_go'] == 'T'].copy()
        df_undersampled = pd.concat(
            [df_undersampled_F, df_undersampled_T])
        if gpu == '0':
            print('len of undersampled train_df', len(df_undersampled))
        training_df = df_undersampled
    elif label_col_name == 'selected_go_vs':
        def find_go(row, go_id = selected_go):
            if go_id in row.go:
                res = 'is_' + selected_go
            else:
                res = 'not_' + selected_go
            return res
        df['selected_go'] = df.apply(find_go, axis=1, go_id = selected_go)
        df['selected_go2'] = df.apply(find_go, axis=1, go_id = selected_go2)
        available_T1 = (df['selected_go'] == 'is_' + selected_go).sum()
        available_T2 = (df['selected_go2'] == 'is_' + selected_go2).sum()
        if gpu == 0:
            print('number of rows that has', selected_go, 'is: ', available_T1)
            print('number of rows that has', selected_go, 'is: ', available_T2)

        df_undersampled_1 = df[df['selected_go'] == 'is_' + selected_go & df['selected_go2'] == 'not_' + selected_go2].copy()
        df_undersampled_2 = df[df['selected_go_2'] == 'is_' + selected_go2 & df['selected_go'] == 'not_' + selected_go].copy()
        df_undersampled = pd.concat(
            [df_undersampled_1, df_undersampled_2])
        if gpu == '0':
            print('len of undersampled train_df', len(df_undersampled))
        training_df = df_undersampled
    else:
        training_df = df
#%%
    if vocab is not None:
        vocab_obj = pickle.load(open(vocab, 'rb'))
        if gpu == 0:
            print('vocab object loaded, len', len(vocab_obj))
    else:
        vocab_obj = None
    if use_sp_processor: # './data/sprot_lm/tmp/spm.model', './data/sprot_lm/tmp/spm.vocab'
        processor = [OpenFileProcessor(), SPProcessor(sp_model=sp_model, sp_vocab=sp_vocab, max_sentence_len=35826, max_vocab_sz=max_vocab)]
    else:
        """## Tokenization"""
        tokenizer = Tokenizer(tok_func=dna_tokenizer, pre_rules=[],
                            post_rules=[], special_cases=[])
        if vocab is not None:
            vocab_class_obj = Vocab.load(vocab)
        else:
            vocab_class_obj = None
        processor = [TokenizeProcessor(tokenizer=tokenizer, include_bos=True,
                                       include_eos=True), NumericalizeProcessor(vocab=vocab_class_obj, max_vocab=max_vocab)]
    # import pdb; pdb.set_trace()
    data_cls = (TextList.from_df(training_df, path=local_project_path, cols=sequence_col_name, processor=processor, vocab=vocab_obj)
                    .split_by_rand_pct(0.1, seed = random_seed)
                    .label_from_df(cols=label_col_name)
                    .databunch(bs=bs, num_workers=workers))
    
    # data_cls.show_batch()
    if gpu == 0:
        print('sample x ', data_cls.train_ds.x[0].text)
        print('sample y ', data_cls.train_ds.y[0])

#%%
    if vocab is None:
        data_cls.vocab.save(local_project_path +
                       'vocab_cls-' + datetime_str + '.pickle')
    if gpu == 0:
        print('data_cls Training set size', len(data_cls.train_ds))
        print('data_cls Validation set size', len(data_cls.valid_ds))
        print('vocab size ', len(data_cls.vocab.itos))

    from sklearn.metrics import f1_score

    @np_func
    def f1(inp, targ):
        return f1_score(targ, np.argmax(inp, axis=-1))

    f1 = FBeta(average='macro')
    f1.beta = 1
#%%
    learn_cls = text_classifier_learner(
        data_cls, eval(network), drop_mult=0.1, pretrained=False,
        metrics=[accuracy, f1], callback_fns=[partial(CSVLogger, append=True), KillerCallback])

    if gpu is None:
        print(gpu, 'DataParallel')
        learn_cls.model = nn.DataParallel(learn_cls.model)
    elif n_gpus > 1:
        print(gpu, 'to_distributed')
        learn_cls.to_distributed(gpu)
        if fp16:
            learn_cls.to_fp16()
    if lm_encoder is not None:
        learn_cls.load_encoder(lm_encoder)
    
    lr = 2e-2
    print(gpu, 'freeze')
    learn_cls.freeze()
    learn_cls.fit_one_cycle(4, lr, moms=(0.8, 0.7))
    
    if benchmarking:
        return
    learn_cls.save('cls-v1-0-' + datetime_str)

    print(gpu, 'unfreeze')
    learn_cls.freeze_to(-2)
    learn_cls.fit_one_cycle(2, slice(lr/(2.6**4), lr), moms=(0.8, 0.7))
    learn_cls.save('cls-v1-1-' + datetime_str)

    learn_cls.freeze_to(-3)
    learn_cls.fit_one_cycle(2, slice(lr/2/(2.6**4), lr/2), moms=(0.8, 0.7))
    learn_cls.save('cls-v1-2-' + datetime_str)
    
    learn_cls.unfreeze()
    learn_cls.fit_one_cycle(4, slice(lr/10/(2.6**4), lr/10), moms=(0.8, 0.7))
    learn_cls.save('cls-v1-3-' + datetime_str)

    learn_cls.fit_one_cycle(20, slice(lr/10/(2.6**4), lr/10), moms=(0.8, 0.7))
    learn_cls.save('cls-v1-4-' + datetime_str)
    
    # learn_cls.export(file = 'export-cls-v1-4-' + datetime_str+ '.pkl')
    print('Done')

# main(None)


#%%
