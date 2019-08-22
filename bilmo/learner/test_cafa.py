from bilmo.scripts.config import Config
from fastai.basic_data import DatasetType
import pandas as pd
from goatools import obo_parser
import logging

conf = Config.conf
log = logging.getLogger("cafa-logger")
obo_go_path = conf['data_path'] + 'cafa3/Gene Ontology Hirarchy/' + \
        'gene_ontology_edit.obo.2016-06-01'
p_go = obo_parser.GODag(obo_go_path)


def write_result(uniq_ids, preds, path, prediction_classes):
    header = ('AUTHOR ARODZ\n' + 'MODEL 1\n' + 'KEYWORDS machine learning\n')
    with open(path, 'a') as file:
        file.write(header)
        log.info('number of prediction_classes: ' + str(len(prediction_classes)))
        leaf_counter = 0
        for i, go_id in enumerate(prediction_classes):
            if go_id == 'others':
                continue
            if len(p_go[go_id].children) != 0: # not a leaf
                continue
            leaf_counter += 1
            df = pd.DataFrame(
                {'cafa_id': uniq_ids, 'go_id': go_id, 'preds': preds[:, i]})
            df['preds'] = df.apply(lambda r: "{:0.2f}".format(r.preds), axis=1)
            file.write(df.to_csv(index=False, header=False, sep='\t'))
        log.info('leaf_counter: ' + str(leaf_counter))
        file.write('END')



def test_cafa(data_test, learn_cls, df_test):
    learn_cls.data.add_test(data_test.train_ds)
    preds, _ = learn_cls.get_preds(ds_type=DatasetType.Test, ordered=True)

    write_result(df_test.uniq_id, preds, conf['local_project_path'] +
                 'result/' + 'result-' + conf['datetime'] + '.txt', learn_cls.data.classes)
