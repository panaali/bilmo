from datetime import datetime

class Config:
    conf = {
        'log_level': 'DEBUG',
        'log_path' : './log/',
        'datetime' : f'{datetime.now():%Y-%m-%d_%H-%M-%S%z}',
        'data_path' : './data/',
        'use_cached_data_cls' : True,
        'bs' : 32,
        'val_bs': 64,
        'n_workers' : 1,
        'fp16' : False,
        'tokenizer_number_of_char' : 1,
        'use_sentencePiece': False,
        'sentencePiece' : {
            'model_path': 'tmp/spm.model', # None to train a new sentencepiece
            'vocab': 'tmp/spm.vocab',  # None to train a new sentencepiece
            'max_vocab' : 30000,
            'max_sentence_len' : 40000
        },
        'vocab_path': 'vocab_lm_sproat_seq_anc_tax.pickle',
        'lm_encoder_path': 'lm-sp-ans-v1-5-enc',
        'gpu' : 0,
        'training_dataframe_path': './data/cafa3/CAFA 3 Protein Targets/CAFA3_training_data/cafa_train_enhanced.p',
        'sequence_col_name' : 'sequence', # seq_tax (ToDo), seq_tax_anc
        'classificiation_type': 'binary',  # binary, multiclass, multilabel
        'binary' : {    # for one vs rest
            'selected_class': 'GO:0036094'
        },
        'multiclass' : {
            'use_top_n_class' : True,
            'n' : 10,
            'selected_classes': ['GO:0036094', 'GO:0036093'],   # Could be used as binary one vs another
        },
        'multilabel' : {

        },
        'sampling_method' : 'undersampling',
        'valid_split_percentage' : 0.1,
        'network' : 'my_AWD_LSTM', # Transformer, TransformerXL
        'random_seed' : 42,
        'max_vocab' : 60000,
        'smaller_valid_df': 600,  # `None` to use full validation
    }
    conf['log_filename']= conf['datetime']
    conf['local_project_path'] = conf['data_path'] + 'sprot_lm/'
    conf['data_cls_cached_location'] = conf['local_project_path'] + \
        'data_cls/data_cls.pickle'
    @classmethod
    def get(cls, parameter):
        if parameter in cls.conf:
            return cls.conf[parameter]
        else:
            raise "config not set inside the config.py file"
