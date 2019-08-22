from bilmo.scripts.config import Config
from bilmo.dataset.tokenizer import dna_tokenizer_n_char
from fastai.text.data import TokenizeProcessor, NumericalizeProcessor, OpenFileProcessor, SPProcessor, TextDataBunch, TextClasDataBunch, TextList
from fastai.text.transform import BaseTokenizer, Vocab
from fastai.text import Tokenizer
import pandas as pd
import os.path
from hashlib import md5

import logging
import dill as pickle

conf = Config.conf
log = logging.getLogger("cafa-logger")
def get_vocab():
    if conf['vocab_path'] is not None:
        vocab_obj = pickle.load(
            open(conf['local_project_path'] + conf['vocab_path'], 'rb'))
        vocab_class_obj = Vocab.load(
            conf['local_project_path'] + conf['vocab_path'])
        log.debug('vocab object loaded, len ' + str(len(vocab_obj)))
    else:
        vocab_obj = None
        vocab_class_obj = None
    return vocab_obj, vocab_class_obj


def get_processor(vocab_class_obj):
    if conf['use_sentencePiece']:
        processor = [OpenFileProcessor(), SPProcessor(
            sp_model=conf['sentencePiece']['model'], sp_vocab=conf['sentencePiece']['vocab'], max_sentence_len=conf['sentencePiece']['max_sentence_len'], max_vocab_sz=conf['sentencePiece']['max_vocab'])]
    else:
        tokenizer = Tokenizer(tok_func=dna_tokenizer_n_char, pre_rules=[],
                              post_rules=[], special_cases=[])
        processor = [TokenizeProcessor(tokenizer=tokenizer, include_bos=True,
                                       include_eos=True), NumericalizeProcessor(vocab=vocab_class_obj, max_vocab=conf['max_vocab'])]
    return processor, tokenizer


def get_cached_data():
    location = get_data_location()
    if conf['use_cached_data_cls'] and os.path.exists(location):
        data_cls, data_test= pickle.load(open(location, 'rb'))
        print_data_cls_info(data_cls)
        print_data_test_info(data_test)
        return data_cls, data_test
    return None, None

def get_hash_data_cache():
    conf_str = (str(conf['bs']) + str(conf['val_bs']) +
                str(conf['vocab_path']) + str(conf['classificiation_type']) +
                str(conf['binary']) + str(conf['multiclass']) +
                str(conf['use_sentencePiece']) +
                str(conf['tokenizer_number_of_char']) +
                str(conf['sequence_col_name']) +
                str(conf['max_vocab']) +
                str(conf['n_workers']) +
                str(conf['valid_split_percentage']))
    conf_str = conf_str.encode('utf-8')
    hash_str = md5(conf_str).hexdigest()
    log.debug('cached data_class conf: ' + str(conf_str))
    log.debug('cached data_class hash: ' + hash_str)
    return hash_str
def get_data_location():
    return conf['data_cached_location'] + get_hash_data_cache() + '.pickle'
def save_data(data_cls, data_test):
    if conf['data_cached_location']:
        pickle.dump((data_cls, data_test), open(get_data_location(), 'wb'))


def save_vocab(data_cls):
    if conf['vocab_path'] is None:
        data_cls.vocab.save(conf['local_project_path'] + 'vocab/' +
                            'vocab_cls-' + conf['datetime'] + '.pickle')

def print_data_cls_info(data_cls):
    log.debug('sample x: ' + data_cls.train_ds.x[0].text)
    log.debug('sample y: ' + str(data_cls.train_ds.y[0]))
    log.debug('data_cls Training set size: ' + str(len(data_cls.train_ds)))
    log.debug('data_cls Validation set size: ' + str(len(data_cls.valid_ds)))
    log.debug('vocab size: ' + str(len(data_cls.vocab.itos)))

def print_data_test_info(data_test):
    log.debug('data_test (cafa3 targets) size: ' + str(len(data_test.train_ds)))


def create_databunch(train_df, valid_df, df_test):
    vocab_obj, vocab_class_obj = get_vocab()
    processor, tokenizer = get_processor(vocab_class_obj)

    if conf['smaller_valid_df'] is not None:
        valid_df_small = valid_df[:conf['smaller_valid_df']]
    else:
        valid_df_small = valid_df

    if conf['classificiation_type'] == 'multilabel':
        label_delim = ' '
    else:
        label_delim = None
    data_cls = TextClasDataBunch.from_df(conf['local_project_path'],
                                         train_df=train_df,
                                         valid_df=valid_df_small,
                                         test_df=valid_df,
                                         tokenizer=tokenizer,
                                         vocab=vocab_class_obj,
                                         text_cols=conf['sequence_col_name'],
                                         label_cols=conf['class_col_name'],
                                         label_delim=label_delim,
                                         max_vocab=conf['max_vocab'],
                                         min_freq=2,
                                         include_bos=True,
                                         include_eos=True,
                                         bs=conf['bs'],
                                         val_bs=conf['val_bs'],
                                         num_workers=conf['n_workers'])

    print_data_cls_info(data_cls)


    data_test = (TextList.from_df(df_test, path=conf['local_project_path'], cols=conf['sequence_col_name'], processor=processor, vocab=data_cls.vocab)
                .split_none()
                .label_empty()
                .databunch(bs=conf['bs'], val_bs=conf['val_bs'], num_workers=conf['n_workers']))

    print_data_test_info(data_test)

    save_data(data_cls, data_test)
    save_vocab(data_cls)
    return data_cls, data_test
