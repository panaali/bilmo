from src.scripts.config import Config
import logging
import pickle 
import numpy as np
import pandas as pd
conf = Config.conf
log = logging.getLogger("cafa-logger")

def prepare_training_binary(df):
    selected_class = conf['binary']['selected_class']
    def find_go(row):
        if selected_class in row.go:
            res = selected_class
        else:
            res = 'others'
        return res
    df['selected_class'] = df.apply(find_go, axis=1)
    available_T = (df['selected_class'] == selected_class).sum()
    available_F = len(df) - available_T
    log.debug('number of rows that has' + selected_class + 'is: ' + str(available_T))

    df_F = df[df['selected_class'] == 'others']
    df_T = df[df['selected_class'] == selected_class]

    valid_len_T = int(available_T * conf['valid_split_percentage'])
    train_len_T = available_T - valid_len_T
    train_len_F = train_len_T
    valid_len_F = available_F - train_len_F
    log.debug('train_len_T ' + str(train_len_T) + ' train_len_F ' + str(train_len_F) + ' valid_len_T ' +
                str(valid_len_T) + ' valid_len_F ' + str(valid_len_F))
    idx_T = np.random.permutation(range(available_T))
    idx_F = np.random.permutation(range(available_F))

    train_T = df_T.iloc[idx_T][:train_len_T]
    valid_T = df_T.iloc[idx_T][train_len_T:]

    train_F = df_F.iloc[idx_F][:train_len_F]
    valid_F = df_F.iloc[idx_F][train_len_F:]

    df_train = pd.concat([train_T, train_F])
    df_valid = pd.concat([valid_T, valid_F])
    return df_train, df_valid



def prepare_training_multiclass(df):
    selected_classes = conf['multiclass']['selected_classes']
    selected_class = selected_classes[0]
    selected_class2 = selected_classes[1]
    def find_go(row, go_id=selected_class):
        if go_id in row.go:
            res = 'is_' + selected_class
        else:
            res = 'not_' + selected_class
        return res
    df['selected_class'] = df.apply(find_go, axis=1, go_id=selected_class)
    df['selected_class2'] = df.apply(find_go, axis=1, go_id=selected_class2)
    available_T1 = (df['selected_class'] == 'is_' + selected_class).sum()
    available_T2 = (df['selected_class2'] == 'is_' + selected_class2).sum()

    log.debug('number of rows that has ' + selected_class + 'is: ' + str(available_T1))
    log.debug('number of rows that has ' + selected_class + 'is: ' + str(available_T2))

    df_undersampled_1 = df[df['selected_class'] == 'is_' + selected_class &
                           df['selected_class2'] == 'not_' + selected_class2].copy()
    df_undersampled_2 = df[df['selected_class_2'] == 'is_' +
                           selected_class2 & df['selected_class'] == 'not_' + selected_class].copy()
    df_undersampled = pd.concat(
        [df_undersampled_1, df_undersampled_2])
    log.debug('len of undersampled train_df ' + str(len(df_undersampled)))
    return df_undersampled

def prepare_training_df(df):
    df = df.dropna(subset=[conf['sequence_col_name']]).copy()
    log.info('total number of rows after removing NaN ' + str(len(df)))
    if conf['classificiation_type'] == 'binary':
        return prepare_training_binary(df)
    elif conf['classificiation_type'] == 'multiclass':
        return prepare_training_multiclass(df)


def load_data():
    if conf['training_dataframe_path'] == None:
        raise BaseException("training_dataframe_path not set in config file")
    df = pickle.load(open(conf['training_dataframe_path'], 'rb'))
    log.debug(df.columns)
    log.info('total number of rows ' + str(len(df)))
    df_train, df_valid = prepare_training_df(df)
    return df_train, df_valid
