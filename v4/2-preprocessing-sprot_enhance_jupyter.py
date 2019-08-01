#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), '../cafa/v4'))
	print(os.getcwd())
except:
	pass


from goatools import obo_parser
import pandas as pd
import pickle
from Bio import SeqIO
sproat_pickle_file = '../data/uniprot_sprot/sproat_parser_df_sp.p'
sprot_parser_df = pickle.load(open(sproat_pickle_file, 'rb'))
print('len sprot_parser_df', len(sprot_parser_df))

obo_tax_file = 'ncbitaxon.obo'
obo_tax_path = '../data/cafa3/Gene Ontology Hirarchy/'
p_tax = obo_parser.GODag(obo_tax_path + obo_tax_file)

print(p_tax)

#%%
def add_Tax_ancesstors(row):
    taxon_name = 'NCBITaxon:' + str(row.taxonomy_id[0])
    try :
        return p_tax[taxon_name].get_all_parents()
    except:
        print(f"errror in fetching {row.taxonomy_id}")
    return None
sprot_parser_df['Tax_ancestors'] = sprot_parser_df.apply(add_Tax_ancesstors, axis=1)

#%%
sample_tax = sprot_parser_df.iloc[0].taxonomy_id[0]
ancestors = [x.replace('NCBITaxon:', '') for x in p_tax['NCBITaxon:' + sample_tax].get_all_parents()]
ancestors.append(sample_tax)

def add_tax_anc_to_seq(row):
	tax_id = row.taxonomy_id[0]
	ancestors = [x.replace('NCBITaxon:', '') for x in p_tax['NCBITaxon:' + tax_id].get_all_parents()]
	ancestors.append(tax_id)
	return ' ' + row.sequence + ' '.join(ancestors)

sprot_parser_df['seq_anc_tax'] = sprot_parser_df.apply(
	add_tax_anc_to_seq, axis=1)



#%%
ali =2
ali

#%%
ali
