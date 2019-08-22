#!/usr/bin/env python
# coding: utf-8
"""
Convert CAFA3 target files to dataframe
"""
# %%
import pandas as pd
from glob import glob
from Bio import SeqIO
import os 
# %%
data_path = './data/'
target_path = data_path + 'cafa3/CAFA 3 Protein Targets/CAFA3_targets/Target files/'
col_ids = []
prot_names = []
col_seqs = []
taxon_ids = []
for file in glob(target_path + "*.fasta"):
    taxon_id = os.path.basename(file).split('.')[1]
    for seq_record in SeqIO.parse(file, "fasta"):
        col_ids.append(seq_record.id)
        desc = seq_record.description.split(' ')
        protein_name = desc[1] if len(desc) > 1 else None
        prot_names.append(protein_name)
        col_seqs.append(str(seq_record.seq))
        taxon_ids.append(taxon_id)
cols={'uniq_id': col_ids, 'sequence': col_seqs, 'prot_names': prot_names, 'taxon_id' : taxon_ids}

# %%
df = pd.DataFrame(data=cols)
print('len(df)', len(df))
# %%
df.to_csv(data_path + 'cafa3/targets.csv', index=False)





#%%
