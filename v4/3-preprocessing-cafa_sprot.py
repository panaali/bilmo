#!/usr/bin/env python
# coding: utf-8
"""
Add taxonomy ancestors to the sprot
"""
#%%
from goatools import obo_parser
from Bio import SeqIO
import pandas as pd
import pickle

data_path = '../cafa/data/'

#%%


def get_cafa_train_df(file_path):
    col_ids = []
    col_seqs = []
    for seq_record in SeqIO.parse(file_path, "fasta"):
        col_ids.append(seq_record.id)
        col_seqs.append(str(seq_record.seq))
    cols = {'accession': col_ids, 'sequence': col_seqs}
    df_cafa_train = pd.DataFrame(data=cols)
    df_cafa_train.set_index('accession', inplace=True)
    return df_cafa_train

#%%


def get_cafa_train_df_gos(file_path):
    df_go = pd.read_csv(file_path, delimiter='\t',
                        header=None, names=['accession', 'GO', 'domain'])
    df_go.set_index('accession', inplace=True)
    return df_go

#%%


def get_not_IEA(df):
    def not_IEA(row):
        # TODO: Fix this
        # even it has one GO with IEA it will remove it
        for x in row:
            if x[0] == 'GO' and 'IEA' not in x[3]:
                return True
        return False

    result = df[df['cross_references'].apply(not_IEA)]
    return result

#%%


def seperate_domains_gos(df_cafa_train_gos):
    df_cafa_train_gos_F = df_cafa_train_gos[df_cafa_train_gos['domain'] == 'F']
    df_cafa_train_gos_P = df_cafa_train_gos[df_cafa_train_gos['domain'] == 'P']
    df_cafa_train_gos_C = df_cafa_train_gos[df_cafa_train_gos['domain'] == 'C']

    df_cafa_train_gos_agg = df_cafa_train_gos.groupby(
        'accession').aggregate(lambda s: set(s))
    df_cafa_train_gos_agg.rename(
        columns={"GO": "gos_leaf", "domain": "gos_leaf_domain"}, inplace=True)

    df_cafa_train_gos_agg_F = df_cafa_train_gos_F.groupby(
        'accession').aggregate(lambda s: set(s)).rename(columns={"GO": "gos_leaf_F", "domain": "domain_leaf_F"})
    df_cafa_train_gos_agg_P = df_cafa_train_gos_P.groupby(
        'accession').aggregate(lambda s: set(s)).rename(columns={"GO": "gos_leaf_P", "domain": "domain_leaf_P"})
    df_cafa_train_gos_agg_C = df_cafa_train_gos_C.groupby(
        'accession').aggregate(lambda s: set(s)).rename(columns={"GO": "gos_leaf_C", "domain": "domain_leaf_C"})

    df_cafa_train_gos_agg_concat = pd.concat([df_cafa_train_gos_agg, df_cafa_train_gos_agg_F,
                                              df_cafa_train_gos_agg_P, df_cafa_train_gos_agg_C], axis=1, sort=False)

    df_cafa_train_gos_agg_concat.drop(
        ['domain_leaf_F', 'domain_leaf_P', 'domain_leaf_C', 'gos_leaf_domain'], axis=1, inplace=True)

    return df_cafa_train_gos_agg_concat


#%%
if __name__ == "__main__":
    sprot_pickle_file = data_path + 'uniprot_sprot/sprot_with_tax_anc_df.p'
    cafa_go_path = data_path + \
        'cafa3/CAFA 3 Protein Targets/CAFA3_training_data/' + 'uniprot_sprot_exp.txt'
    cafa_train_fasta_path = data_path + \
        'cafa3/CAFA 3 Protein Targets/CAFA3_training_data/' + 'uniprot_sprot_exp.fasta'
    obo_go_path = data_path + 'cafa3/Gene Ontology Hirarchy/' + \
        'gene_ontology_edit.obo.2016-06-01'
    cafa_train_enhanced = data_path + \
        'cafa3/CAFA 3 Protein Targets/CAFA3_training_data/cafa_train_enhanced.p'
#%%
    sprot_parser_df = pickle.load(open(sprot_pickle_file, 'rb'))
    print('len sprot_parser_df', len(sprot_parser_df))
#%%
    # df_cafa_train_seq = get_cafa_train_df(cafa_train_fasta_path)
#%%
    df_cafa_train_gos = get_cafa_train_df_gos(cafa_go_path)
    df_cafa_train_gos_seperated = seperate_domains_gos(df_cafa_train_gos)
#%%
    sprot_parser_df = sprot_parser_df.set_index('primary_accession')
#%%

    def get_seq_anc(row):
        try:
            return sprot_parser_df.loc[row.name, 'seq_anc_tax']
        except:
            try:
                return sprot_parser_df[[
                    row.name in s for s in sprot_parser_df.accessions]].iloc[0].seq_anc_tax
            except:
                return None
    df_cafa_train_gos_seperated['seq_anc_tax'] = df_cafa_train_gos_seperated.apply(
        get_seq_anc, axis=1)

    def get_seq(row):
        try:
            return sprot_parser_df.loc[row.name, 'sequence']
        except:
            try:
                return sprot_parser_df[[
                    row.name in s for s in sprot_parser_df.accessions]].iloc[0].seq_anc_tax
            except:
                return None
    df_cafa_train_gos_seperated['sequence'] = df_cafa_train_gos_seperated.apply(
        get_seq, axis=1)
#%%
    # Add go_ancs
    p_go = obo_parser.GODag(obo_go_path)

#%%
    def add_GO_ancesstors(row, row_namespace):
        go_ans = set()
        go_list = row.gos_leaf
        for go in go_list:
            if row_namespace == 'all' or p_go[go].namespace == row_namespace:
                go_ans.add(go)
                try:
                    go_ans |= p_go[go].get_all_parents()
                except:
                    print(f"errror in fetching {go}")
        return go_ans
#%%
    df_cafa_train_gos_seperated['go_C'] = df_cafa_train_gos_seperated.apply(
        add_GO_ancesstors, axis=1, row_namespace='cellular_component')
#%%
    df_cafa_train_gos_seperated['go_P'] = df_cafa_train_gos_seperated.apply(
        add_GO_ancesstors, axis=1, row_namespace=('biological_process'))
#%%
    df_cafa_train_gos_seperated['go_F'] = df_cafa_train_gos_seperated.apply(
        add_GO_ancesstors, axis=1, row_namespace=('molecular_function'))

#%%
    df_cafa_train_gos_seperated['go'] = df_cafa_train_gos_seperated.apply(
        add_GO_ancesstors, axis=1, row_namespace=('all'))


#%%
    pickle.dump(df_cafa_train_gos_seperated, open(cafa_train_enhanced, 'wb'))

#%%
