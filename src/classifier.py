
#%%
from textwrap import wrap

from fastai.text.models import AWD_LSTM, TransformerXL, Transformer, Activation
def run_from_ipython():
    try:
        __IPYTHON__
        return True
    except NameError:
        return False

#%%
from fastai.distributed import setup_distrib
from fastai.text import Tokenizer
from fastai.text.data import TokenizeProcessor, NumericalizeProcessor, OpenFileProcessor, SPProcessor, TextDataBunch, TextClasDataBunch
from fastai.text.data import TextList
from fastai.text.learner import text_classifier_learner
from fastai.text.transform import BaseTokenizer, Vocab
from fastai.torch_core import num_distrib, np_func
from fastai.core import num_cpus
from fastai.script import call_parse, Param
from fastai.metrics import accuracy, FBeta
from fastai.callbacks.csv_logger import CSVLogger
from fastai.basic_data import DatasetType
from fastprogress import fastprogress
from functools import partial
from datetime import datetime
import pickle
import numpy as np
import os
from torch import nn
import torch
import pandas as pd
import inspect


if run_from_ipython():
    from src.callbacks import *
else:
    from callbacks import *

"""
Classifier
"""
# %%
data_path = './data/'

class dna_tokenizer_n_char(BaseTokenizer):
    n_char = 1
    def tokenizer(self, t):
        tokens = t.split(' ')
        bos = tokens[0]
        seq = tokens[1]
        after_seq = tokens[2:-1]
        eos = tokens[-1]
        result = [bos]
        result += wrap(seq, self.n_char)  # sequence string to list
        result += after_seq
        result.append(eos)
        return result

# %%
def make_training_selected_go(label_col_name, df, selected_go, selected_go2, gpu, valid_split_percentage):
    def find_go(row):
        if selected_go in row.go:
            res = selected_go
        else:
            res = 'others'
        return res
    df['selected_go'] = df.apply(find_go, axis=1)
    available_T = (df['selected_go'] == selected_go).sum()
    available_F = len(df) - available_T
    if gpu == 0:
        print('number of rows that has', selected_go, 'is: ', available_T)
    
    df_F = df[df['selected_go'] == 'others']
    df_T = df[df['selected_go'] == selected_go]

    valid_len_T = int(available_T * valid_split_percentage)
    train_len_T = available_T - valid_len_T
    train_len_F = train_len_T
    valid_len_F = available_F - train_len_F
    if gpu == 0:
        print('train_len_T', train_len_T, 'train_len_F', train_len_F, 'valid_len_T',
              valid_len_T, 'valid_len_F', valid_len_F)
    idx_T = np.random.permutation(range(available_T))
    idx_F = np.random.permutation(range(available_F))

    train_T = df_T.iloc[idx_T][:train_len_T]
    valid_T = df_T.iloc[idx_T][train_len_T:]

    train_F = df_F.iloc[idx_F][:train_len_F]
    valid_F = df_F.iloc[idx_F][train_len_F:]

    df_train = pd.concat([train_T, train_F])
    df_valid = pd.concat([valid_T, valid_F])
    return df_train, df_valid


# %%
def make_training_selected_go_vs(label_col_name, df, selected_go, selected_go2, gpu, valid_split_percentage):
    def find_go(row, go_id=selected_go):
        if go_id in row.go:
            res = 'is_' + selected_go
        else:
            res = 'not_' + selected_go
        return res
    df['selected_go'] = df.apply(find_go, axis=1, go_id=selected_go)
    df['selected_go2'] = df.apply(find_go, axis=1, go_id=selected_go2)
    available_T1 = (df['selected_go'] == 'is_' + selected_go).sum()
    available_T2 = (df['selected_go2'] == 'is_' + selected_go2).sum()
    if gpu == 0:
        print('number of rows that has', selected_go, 'is: ', available_T1)
        print('number of rows that has', selected_go, 'is: ', available_T2)

    df_undersampled_1 = df[df['selected_go'] == 'is_' + selected_go &
                            df['selected_go2'] == 'not_' + selected_go2].copy()
    df_undersampled_2 = df[df['selected_go_2'] == 'is_' +
                            selected_go2 & df['selected_go'] == 'not_' + selected_go].copy()
    df_undersampled = pd.concat(
        [df_undersampled_1, df_undersampled_2])
    if gpu == 0:
        print('len of undersampled train_df', len(df_undersampled))
    return df_undersampled

# %%
def make_training_df(label_col_name, df, selected_go, selected_go2, gpu, valid_split_percentage):
    df = df.dropna(subset=['seq_anc_tax'])
    print('total number of rows after removing Nan', len(df))
    if label_col_name == 'selected_go':
        return make_training_selected_go(label_col_name, df, selected_go, selected_go2, gpu, valid_split_percentage)
    elif label_col_name == 'selected_go_vs':
        return make_training_selected_go_vs(label_col_name, df, selected_go, selected_go2, gpu, valid_split_percentage)
        
# %%


def write_result(uniq_ids, preds, go_ids, path, prediction_classes):
    header = ('AUTHOR ARODZ\n' +
                'MODEL 1\n' +
                'KEYWORDS Neural Networks, Language Modeling\n')
    with open(path, 'a') as file:
        file.write(header)
        for i, go_id in enumerate(prediction_classes):
            df = pd.DataFrame(
                {'cafa_id': uniq_ids, 'go_id': go_id, 'preds': preds[:,i]})
            file.write(df.to_csv(index=False, header= False, sep='\t'))
# %%
@call_parse
def main(train_df_path: Param("location of the training dataframe", str, opt=False),
         gpu: Param("Passed by fastai.launch", str) = None,
         local_rank: Param('Passed by torch.launch', str) = None,
         max_cpu_per_dataloader: Param("Max CPU", int) = 8,
         bs: Param("batch size", int) = 64,
         val_bs: Param('validation batch size', int) = 128,
         fp16: Param("mixed precision", int) = 0,
         use_sp_processor: Param("use sentence piece as processor", int) = 0,
         sp_model: Param("sentence piece trained model file", str) = None,
         sp_vocab: Param("sentence piece trained vocab file", str) = None,
         lm_encoder: Param("language modeling encoder file", str) = None,
         sequence_col_name: Param("name of the sequence column",
                                  str) = 'seq_anc_tax',
         label_col_name: Param('label column name', str) = 'selected_go',
         selected_go: Param('which Go_id for binary classfication. selected_go vs others', str) = None,
         selected_go2: Param('which Go_id for binary classfication. selected_go vs selected_go2', str) = None,
         vocab_path: Param('vocab file', str) = None,
         benchmarking: Param('benchmarking', int) = 0,
         tokenizer_n_char: Param('every N chararachters for tokenization', int) = 1,
         valid_split_percentage: Param('Validation split percentage', str) = '0.1',
        #  skip: Param('Skip training except first epoch', int) = 0,
         network: Param(
             'Which network to use? AWD_LSTM, Transformer, TransformerXL', str) = 'AWD_LSTM'
         ):
    
# %%
    datetime_str = f'{datetime.now():%Y-%m-%d_%H-%M-%S%z}'
    data_path = '../cafa/data/'
# %%
    # # For iPython testing only
    
    max_cpu_per_dataloader = 0
    bs = 64
    val_bs = 128
    fp16 = 0
    use_sp_processor = 0
    sp_model = None
    sp_vocab = None
    gpu = 0
    local_rank = None
    train_df_path = data_path + \
        'cafa3/CAFA 3 Protein Targets/CAFA3_training_data/cafa_train_enhanced.p'
    sequence_col_name = 'sequence'
    label_col_name = 'selected_go'
    selected_go = 'GO:0036094'
    # vocab_path = data_path + 'sprot_lm/vocab_lm_sproat_seq_anc_tax.pickle'
    # lm_encoder = 'lm-sp-ans-v1-5-enc'
    vocab_path = None
    lm_encoder = None
    network = 'AWD_LSTM'
    tokenizer_n_char = 1
    gpus = '0'
    gpus = list(range(torch.cuda.device_count())) if gpus=='all' else list(gpus)
    os.environ["WORLD_SIZE"] = '0'
    valid_split_percentage = 0.1
    selected_go2 = None
    skip = 1

# %%
    valid_split_percentage = float(valid_split_percentage)
    fastprogress.SAVE_PATH = f'./log/fastprogress-{datetime_str}.txt'
    random_seed = 42
    max_vocab = 60000

    print(datetime_str)
# %%
    arg_strs = '############################################################\n'
    arg_strs += datetime_str + '\n'
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    if gpu == '0':
        print('function name "%s"' % inspect.getframeinfo(frame)[2])
    for i in args:
        arg_str = "    %s = %s" % (i, values[i])
        arg_strs += arg_str + '\n'
        if gpu == '0':
            print(arg_str)
    """## Prepare Dataset"""
    local_project_path = data_path + 'sprot_lm/'
# %%
    # Distributed
    if gpu == None and local_rank != None:
        gpu = local_rank
        torch.distributed.init_process_group(
            backend='nccl', init_method='env://')

    print('gpu', gpu)
    n_gpus = num_distrib()
    gpu = setup_distrib(gpu)
    if n_gpus < 2:
        os.environ["WORLD_SIZE"] = '0' # because the average callback has does not consider the case of 1
    if n_gpus > 0:
        workers = min(max_cpu_per_dataloader, num_cpus()//n_gpus)
    else:
        workers = min(max_cpu_per_dataloader, num_cpus())
    
    # os.environ["OMP_NUM_THREADS"] = str(1)
    # os.environ["MKL_NUM_THREADS"] = str(1)
    # torch.set_num_threads(1)
    print(gpu, 'n_gpus', n_gpus)
    print(gpu, 'workers', workers)

# %%
    """## Prepare path"""
    np.random.seed(random_seed)

    if not os.path.exists(local_project_path):
        os.makedirs(local_project_path)
    if not os.path.exists(local_project_path + 'vocab/'):
        os.makedirs(local_project_path)
    print('local_project_path:', local_project_path)


# %%
    df = pickle.load(open(train_df_path, 'rb'))
    print(df.columns)
    print('total number of rows', len(df))
# %%
    df_train, df_valid = make_training_df(
        label_col_name, df, selected_go, selected_go2, gpu, valid_split_percentage)

# %%
    if vocab_path is not None:
        vocab_obj = pickle.load(open(vocab_path, 'rb'))
        if gpu == 0:
            print('vocab object loaded, len', len(vocab_obj))
    else:
        vocab_obj = None
    if use_sp_processor:  # './data/sprot_lm/tmp/spm.model', './data/sprot_lm/tmp/spm.vocab'
        processor = [OpenFileProcessor(), SPProcessor(
            sp_model=sp_model, sp_vocab=sp_vocab, max_sentence_len=35826, max_vocab_sz=max_vocab)]
    else:
        """## Tokenization"""
        dna_tokenizer_n_char.n_char = tokenizer_n_char
        tokenizer = Tokenizer(tok_func=dna_tokenizer_n_char, pre_rules=[],
                              post_rules=[], special_cases=[])
        if vocab_path is not None:
            vocab_class_obj = Vocab.load(vocab_path)
        else:
            vocab_class_obj = None
        processor = [TokenizeProcessor(tokenizer=tokenizer, include_bos=True,
                                       include_eos=True), NumericalizeProcessor(vocab=vocab_class_obj, max_vocab=max_vocab)]
# %%
    df_valid_small = df_valid[:600]
    # import multiprocessing
    # multiprocessing.set_start_method('spawn', True)
    # os.sched_setaffinity(0, {0})
    data_cls = TextClasDataBunch.from_df(local_project_path, train_df=df_train, valid_df=df_valid_small, test_df=df_valid,
                                        tokenizer=tokenizer, vocab=vocab_class_obj,
                                        text_cols=sequence_col_name, #processor=processor,
                                        label_cols=label_col_name, label_delim=None,
                                        max_vocab=max_vocab, min_freq=2, include_bos=True,
                                        include_eos=True,
                                        bs=bs, val_bs=val_bs, num_workers=workers)

    # data_cls.show_batch()
    if gpu == 0:
        print('sample x ', data_cls.train_ds.x[0].text)
        print('sample y ', data_cls.train_ds.y[0])

# %%
    if vocab_path is None:
        data_cls.vocab.save(local_project_path + 'vocab/' +
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
# %%
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
    
    if network == 'AWD_LSTM':
        network_config = dict(emb_sz=30, n_hid=1152, n_layers=3, pad_token=1, qrnn=False, bidir=False, output_p=0.4,
                              hidden_p=0.3, input_p=0.4, embed_p=0.05, weight_p=0.5)
    elif network == 'Transformer':
        network_config = dict(ctx_len=512, n_layers=12, n_heads=12, d_model=768, d_head=64, d_inner=3072, resid_p=0.1, attn_p=0.1,
                              ff_p=0.1, embed_p=0.1, output_p=0., bias=True, scale=True, act=Activation.GeLU, double_drop=False,
                              init=init_transformer, mask=False)
    elif network == 'TransformerXL':
        network_config = dict(ctx_len=150, n_layers=12, n_heads=10, d_model=410, d_head=41, d_inner=2100, resid_p=0.1, attn_p=0.1,
                              ff_p=0.1, embed_p=0.1, output_p=0.1, bias=False, scale=True, act=Activation.ReLU, double_drop=True,
                              init=init_transformer, mem_len=150, mask=False)
    else:
        raise BaseException("network not defined")

    learn_cls = text_classifier_learner(
        data_cls, eval(network), config=network_config, drop_mult=0.1, pretrained=False,
        metrics=[accuracy, f1], callback_fns=[partial(CSVLogger, append=True), KillerCallback])
    if gpu == 0:
        print(learn_cls.summary())
        print(learn_cls.model)
        csv_file = open(local_project_path + 'history/' + 'history.csv', 'a')
        csv_file.write(arg_strs)
        csv_file.close()
    if gpu is None:
        print(gpu, 'DataParallel')
        learn_cls.model = nn.DataParallel(learn_cls.model)
    elif n_gpus > 1:
        print(gpu, 'to_distributed')
        learn_cls.to_distributed(gpu)
        if fp16:
            learn_cls.to_fp16()
    if lm_encoder is not None and vocab_path is not None:
        learn_cls.load_encoder(lm_encoder)

# %%
    lr = 2e-2
    print(gpu, 'freeze')
    learn_cls.freeze()
    learn_cls.fit_one_cycle(1, lr, moms=(0.8, 0.7))
    # if not skip:
# %%
    learn_cls.fit_one_cycle(4, lr, moms=(0.8, 0.7))
    #     
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
        # learn_cls.export(file = 'export/' + 'export-cls-v1-4-' + datetime_str+ '.pkl')
# %%
    # validate on full valid_ds (named test_dl in here!)
    print('Done Training')
    # print('full valid validation: loss, acc, f_beta', learn_cls.validate(learn_cls.data.test_dl))
    print('Start Test Prediction')
# %%
    # add this as the test_ds
    # do prediction and save it with approperiate format
    # ALI: Test result for benchmark
    # TOM: Test Transformer
    # TOM: Test with multiclass / divide all the probablities by the number of true classes.
    # Add https://www.comet.ml/panaali/general/view/
    
    df_test = pd.read_csv(data_path + 'cafa3/targets.csv')
    data_test = (TextList.from_df(df_test[:100], path=local_project_path, cols='sequence', processor=processor, vocab=vocab_obj)
                 .split_none()
                 .label_empty()
                 .databunch(bs=bs, num_workers=workers))
    # data_test = TextClasDataBunch.from_df(local_project_path, train_df=df_train, valid_df=None, test_df=df_test[:100],
    #                                       tokenizer=tokenizer, vocab=vocab_class_obj,
    #                                       text_cols='sequence',  # processor=processor,
    #                                       label_cols=None, label_delim=None,
    #                                       max_vocab=max_vocab, min_freq=2, include_bos=True,
    #                                       include_eos=True,
    #                                       bs=bs, val_bs=val_bs, num_workers=workers)
# %%
    learn_cls.data.add_test(data_test.train_ds)
    preds, _ = learn_cls.get_preds(ds_type=DatasetType.Test, ordered= True)

    write_result(df_test.uniq_id[:100], preds, selected_go, local_project_path +
                 'result/' + 'result-' + datetime_str + '.txt', learn_cls.data.classes)
    
main(None)


# %%
