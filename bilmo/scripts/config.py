from datetime import datetime

class Config:
    conf = {
        'gpu': 1,
        'project_name': 'bilmo-cafa3',
        'use_cached_data_cls': True, # change if error in loading
        'skip_training': False,
        'just_one_epoch': False,
        'use_weight': True,
        'num_epochs': 20,
        'test_on_cafa3_testset': True,
        # if have pretrained language model, put vocab and encoder name in here
        # 'vocab_path': None,
        'vocab_path': 'vocab_lm_sproat_seq_anc_tax.pickle',
        # 'lm_encoder_path': None,
        'lm_encoder_path': 'lm-sp-ans-v1-5-enc',
        # `None` to use full training ,  example: 100
        'smaller_train_df': None,
        # `None` to use full validation , example: 100
        'smaller_valid_df': None,
        # ok for protein centeric evaulation but not for term centeric I guess, otherwise it's 130K
        'predict_only_final_targets': True,
        # MultiLabelCrossEntropy, BCEWithLogitsFlat, BCEWithLogitsLoss
        'loss_func': 'BCEWithLogitsLoss',
        'loss_reduction': 'mean',
        'bs': 32,
        'val_bs': 64,
        # multiprocess for dataloaders or tokenizer. 0 = no multiprocess
        'n_workers': 4,
        'fp16': False,
        'tokenizer_number_of_char': 1,
        'use_sentencePiece': False,
        'sentencePiece': {
            'model_path': 'tmp/spm.model',  # None to train a new sentencepiece
            'vocab': 'tmp/spm.vocab',  # None to train a new sentencepiece
            'max_vocab': 30000,
            'max_sentence_len': 40000
        },
        # sum, mean, none
        'weight_decay': 0.05,  # if underfitting set as 0.01
        'training_dataframe_path':
        './data/cafa3/CAFA 3 Protein Targets/CAFA3_training_data/cafa_train_enhanced.p',
        'sequence_col_name': 'sequence',  # seq_tax (ToDo), seq_tax_anc
        'class_col_name': 'classes',
        'classificiation_type': 'multilabel',  # binary, multiclass, multilabel
        'binary': {  # for one vs rest
            'selected_class': 'GO:0036094'
        },
        'multiclass': {
            'use_top_n_class': True,
            'n': 10,
            'selected_classes':
            ['GO:0036094',
             'GO:0036093'],  # Could be used as binary one vs another
        },
        'multilabel': {},
        'OverSamplingCallback': False,  # doesn't work for multilabel
        'valid_split_percentage': 0.1,
        # my_AWD_LSTM, my_Transformer, my_TransformerXL
        'network': 'my_AWD_LSTM',
        'drop_mult': 1,
        'AWD_STM_config': {
            'emb_sz': 400,
            'n_hid': 1152,
            'n_layers': 3,
            'pad_token': 1,
            'qrnn': False,
            'bidir': False,
            'output_p': 0.4,
            'hidden_p': 0.3,
            'input_p': 0.4,
            'embed_p': 0.05,
            'weight_p': 0.5
        },
        # ctx_len context length for positional encoder
        # d_model the hidden size
        'Transformer_config': {
            'ctx_len': 512,
            'n_layers': 12,
            'n_heads': 12,
            'd_model': 768,
            'd_head': 64,
            'd_inner': 3072,
            'resid_p': 0.1,
            'attn_p': 0.1,
            'ff_p': 0.1,
            'embed_p': 0.1,
            'output_p': 0.,
            'bias': True,
            'scale': True,
            'act': 'Activation.GeLU',
            'double_drop': False,
            'init': 'init_transformer',
            'mask': False
        },
        #mem_len
        'TransformerXL_config': {
            'ctx_len': 150,
            'n_layers': 12,
            'n_heads': 10,
            'd_model': 410,
            'd_head': 41,
            'd_inner': 2100,
            'resid_p': 0.1,
            'attn_p': 0.1,
            'ff_p': 0.1,
            'embed_p': 0.1,
            'output_p': 0.1,
            'bias': False,
            'scale': True,
            'act': 'Activation.ReLU',
            'double_drop': True,
            'init': 'init_transformer',
            'mem_len': 150,
            'mask': False
        },
        'optimizer': 'adam',  # adam, radam # radam seems not good for finetuning
        'random_seed': 42,
        'max_vocab': 60000,
        'export_model': False, #doesn't work?
        'save_model': True,
        'add_tensorboard': True,
        'log_graph_tensorboard': False,
        'cafa_test_top_k_protein': 200,
        'cafa_test_top_k_classes': 200,
        'comet.ml': False,
        'multiLabelF1_thresh': 0.2,  # just for our validation purpose
        'log_level': 'DEBUG',
        'log_path': './log/',
        'datetime': f'{datetime.now():%Y-%m-%d_%H-%M-%S%z}',
        'data_path': './data/',
    }
    conf['log_filename']= conf['datetime']
    conf['local_project_path'] = conf['data_path'] + 'sprot_lm/'
    conf['data_cached_location'] = conf['local_project_path'] + 'data_cls/'
    @classmethod
    def get(cls, parameter):
        if parameter in cls.conf:
            return cls.conf[parameter]
        else:
            raise "config not set inside the config.py file"
