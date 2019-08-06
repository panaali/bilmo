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
#     pickle.dump(list_sp, open( "sprot_parser_list_sp.p", "wb" ) )
    
def to_df():
#     list_sp = pickle.load(open( "sprot_parser_list_sp.p", "rb" ))
    list_sp = parse()
    print(len(list_sp))
    print('list loaded')
    df = pd.DataFrame(list_sp)
    print('df created')
    print(len(df))
    pickle.dump(df, open( "sprot_parser_df_sp.p", "wb" ) )
    print('pickle dumped')
    df.to_feather("sprot_parser_feather_sp.p")
    
if __name__ == '__main__':
    to_df()
