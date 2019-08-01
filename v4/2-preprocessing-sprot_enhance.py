#!/usr/bin/env python
# coding: utf-8
"""
Add taxonomy ancestors to the sproat
"""

from goatools import obo_parser
import pandas as pd
import pickle

data_path = './data/'

def add_tax_ancesstors(row):
    taxon_name = 'NCBITaxon:' + str(row.taxonomy_id[0])
    try:
        return p_tax[taxon_name].get_all_parents()
    except:
        print(f"errror in fetching {row.taxonomy_id}")
    return None


def add_tax_anc_to_seq(row):
	tax_id = row.taxonomy_id[0]
	ancestors = [x.replace('NCBITaxon:', '')
              for x in p_tax['NCBITaxon:' + tax_id].get_all_parents()]
	ancestors.append(tax_id)
	return row.sequence + ' ' + ' '.join(ancestors)


if __name__ == "__main__":
    sproat_pickle_file = data_path + 'uniprot_sprot/sproat_parser_df_sp.p'
    sprot_parser_df = pickle.load(open(sproat_pickle_file, 'rb'))
    print('len sprot_parser_df', len(sprot_parser_df))

    obo_tax_file = data_path + 'cafa3/Gene Ontology Hirarchy/' + 'ncbitaxon.obo'
    p_tax = obo_parser.GODag(obo_tax_file)

    sprot_parser_df['tax_ancestors'] = sprot_parser_df.apply(
        add_tax_ancesstors, axis=1)

    sprot_parser_df['seq_anc_tax'] = sprot_parser_df.apply(
        add_tax_anc_to_seq, axis=1)

    sprot_parser_df_seq = sprot_parser_df[['sequence']]
    sprot_parser_df_seq_anc = sprot_parser_df[['seq_anc_tax']]

    pickle.dump(sprot_parser_df_seq, open(data_path + 'uniprot_sprot/sproat_sequence.pickle', 'wb'))
    pickle.dump(sprot_parser_df_seq_anc,
                open(data_path + 'uniprot_sprot/sproat_sequence_taxon_anc.pickle', 'wb'))
