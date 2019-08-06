#!/usr/bin/env python
# coding: utf-8
"""
Add taxonomy ancestors to the sprot
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
    ancestors.insert(0, tax_id)
    return row.sequence + ' | ' + ' '.join(ancestors)

if __name__ == "__main__":
    sprot_pickle_file = data_path + 'uniprot_sprot/sprot_2017_01.p'
    sprot = pickle.load(open(sprot_pickle_file, 'rb'))
    print('len sprot', len(sprot))

    obo_tax_file = data_path + 'cafa3/Gene Ontology Hirarchy/' + 'ncbitaxon.obo'
    p_tax = obo_parser.GODag(obo_tax_file)

    sprot['tax_ancestors'] = sprot.apply(
        add_tax_ancesstors, axis=1)

    sprot['seq_anc_tax'] = sprot.apply(
        add_tax_anc_to_seq, axis=1)

    sprot_seq = sprot[['sequence']]
    sprot_seq_anc = sprot[['seq_anc_tax']]

    pickle.dump(sprot_seq, open(data_path + 'uniprot_sprot/sprot_sequence.pickle', 'wb'))
    pickle.dump(sprot_seq_anc,
                open(data_path + 'uniprot_sprot/sprot_sequence_taxon_anc.pickle', 'wb'))

    sprot_pickle_file_with_tax_anc = data_path + 'uniprot_sprot/sprot_with_tax_anc.p'
    pickle.dump(sprot, open(sprot_pickle_file_with_tax_anc, 'wb'))
