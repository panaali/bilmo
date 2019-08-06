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

#%%
data_path = '../cafa/data/'


#%%
def get_sprot_gos(row, exclude_IEA = False):
    gos = set()
    for x in row.cross_references:
        if x[0] == 'GO':
            if exclude_IEA:
                if 'IEA' not in x[3]:
                    gos.add(x[1])
            else:
                gos.add(x[1])
    return gos


def get_go_domain(row, p_go = None, row_namespace = 'all', with_ancestors = False, exclude_IEA=False):
    gos_custom = set()
    for go in row.gos_leaf:
        try : 
            if row_namespace == 'all' or p_go[go].namespace == row_namespace:
                gos_custom.add(go)
                if with_ancestors:
                    try:
                        gos_custom |= p_go[go].get_all_parents()
                    except:
                        print(f"Error in fetching {go}")
        except:
            print("Failed to retrieve ", go)
    return gos_custom

#%%
def seperate_gos_by_domains(sprot, p_go):
    # import pdb; pdb.set_trace();
    sprot['gos_leaf_F'] = sprot.apply(
        get_go_domain, axis=1, p_go=p_go, row_namespace='molecular_function')
    sprot['gos_leaf_P'] = sprot.apply(
        get_go_domain, axis=1, p_go=p_go, row_namespace='biological_process')
    sprot['gos_leaf_C'] = sprot.apply(
        get_go_domain, axis=1, p_go=p_go, row_namespace='cellular_component')
    sprot['gos_leaf_not_IEA_F'] = sprot.apply(
        get_go_domain, axis=1, p_go=p_go, exclude_IEA=True, row_namespace='molecular_function')
    sprot['gos_leaf_not_IEA_P'] = sprot.apply(
        get_go_domain, axis=1, p_go=p_go, exclude_IEA=True, row_namespace='biological_process')
    sprot['gos_leaf_not_IEA_C'] = sprot.apply(
        get_go_domain, axis=1, p_go=p_go, exclude_IEA=True, row_namespace='cellular_component')

    sprot['gos_anc'] = sprot.apply(
        get_go_domain, axis=1, p_go=p_go, with_ancestors=True, row_namespace='all')
    sprot['gos_F_anc'] = sprot.apply(
        get_go_domain, axis=1, p_go=p_go, with_ancestors=True, row_namespace='molecular_function')
    sprot['gos_P_anc'] = sprot.apply(
        get_go_domain, axis=1, p_go=p_go, with_ancestors=True, row_namespace='biological_process')
    sprot['gos_C_anc'] = sprot.apply(
        get_go_domain, axis=1, p_go=p_go, with_ancestors=True, row_namespace='cellular_component')
    sprot['gos_not_IEA_F_anc'] = sprot.apply(
        get_go_domain, axis=1, p_go=p_go, with_ancestors=True, exclude_IEA=True, row_namespace='molecular_function')
    sprot['gos_not_IEA_P_anc'] = sprot.apply(
        get_go_domain, axis=1, p_go=p_go, with_ancestors=True, exclude_IEA=True, row_namespace='biological_process')
    sprot['gos_not_IEA_C_anc'] = sprot.apply(
        get_go_domain, axis=1, p_go=p_go, with_ancestors=True, exclude_IEA=True, row_namespace='cellular_component')
    return sprot

#%%
if __name__ == "__main__":
    sprot_pickle_file = data_path + 'uniprot_sprot/sprot_with_tax_anc_df.p'
    obo_go_path = data_path + 'cafa3/Gene Ontology Hirarchy/' + \
        'gene_ontology_edit.obo.2017-02-01'
    sprot_gos_seperated_file = data_path + 'uniprot_sprot/sprot_gos_anc.p'
    sprot_gos_seperated_slimed_file = data_path + 'uniprot_sprot/sprot_gos_anc_slimed.p'
#%%
    sprot = pickle.load(open(sprot_pickle_file, 'rb'))
    print('len sprot', len(sprot))
#%%
    sprot = sprot.set_index('primary_accessiontel')
#%%
    sprot['gos_leaf'] = sprot.apply(get_sprot_gos, axis=1)
    sprot['gos_leaf_not_IEA'] = sprot.apply(
        get_sprot_gos, axis=1, exclude_IEA=True)
#%%
    p_go = obo_parser.GODag(obo_go_path)
#%%
    seperate_gos_by_domains(sprot, p_go)

#%%
    pickle.dump(sprot, open(sprot_gos_seperated_file, 'wb'))
    pickle.dump(sprot[['accessions', 'sequence', 'seq_anc_tax', 'gos_leaf', 'gos_leaf_not_IEA',
                       'gos_leaf_F', 'gos_leaf_P', 'gos_leaf_C', 'gos_leaf_not_IEA_F',
                       'gos_leaf_not_IEA_P', 'gos_leaf_not_IEA_C', 'gos_anc', 'gos_F_anc',
                       'gos_P_anc', 'gos_C_anc', 'gos_not_IEA_F_anc', 'gos_not_IEA_P_anc',
                       'gos_not_IEA_C_anc']], open(sprot_gos_seperated_slimed_file, 'wb'))

#%%
