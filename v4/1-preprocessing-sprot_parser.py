#!/usr/bin/env python
# coding: utf-8
"""
Convert uniprot_sprot.dat file to a pandas dataframe
"""

import pickle
import pandas as pd
from Bio import SwissProt

def parse():
    dat_path = '../../data/uniprot_sprot/uniprot_sprot-only2017_01/'
    dat_file = "uniprot_sprot.dat"

    list_sp = []
    handle = open(dat_path + dat_file)
    i = 0
    for record in SwissProt.parse(handle):
        i += 1
        if i % 1000 == 0:
            print(i)
        list_sp.append(vars(record))

    return list_sp
    
def to_df():
    list_sp = parse()
    print(len(list_sp))
    print('list loaded')
    df = pd.DataFrame(list_sp)
    print('df created')
    print(len(df))
    df['primary_accession'] = df.apply(lambda row: row.accessions[0], axis=1)
    df = df.set_index('primary_accession')

    pickle.dump(df, open("sprot_2017_01.p", "wb"))
    print('pickle dumped')
    
if __name__ == '__main__':
    to_df()
