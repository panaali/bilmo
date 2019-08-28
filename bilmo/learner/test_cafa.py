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
    csv_str = ''
    csv_str += header
    with open(path, 'a') as file:
        log.info('number of prediction_classes: ' + str(len(prediction_classes)))
        top_k_preds = preds.topk(conf['cafa_test_top_k_classes'])
        top_k_preds_probabilities = top_k_preds.values
        top_k_preds_class_ids = top_k_preds.indices
        num_protein = top_k_preds_probabilities.shape[0]
        num_classes = top_k_preds_probabilities.shape[1]
        log.info('total number of protein*top_class: ' + str(num_protein * num_classes))
        sep = '\t'
        counter = 0
        for i_protein in range(num_protein):
            for j_class in range(num_classes):
                counter += 1
                if counter % 10000 == 0:
                    log.info('counter: ' + str(counter))
                uniq_id = uniq_ids.iloc[i_protein]
                class_id = top_k_preds_class_ids[i_protein, j_class]
                probablity = top_k_preds_probabilities[i_protein, j_class].item()
                class_name = prediction_classes[class_id]
                # if len(p_go[class_name].children) != 0:
                #     continue
                csv_str += "{}	{}	{:0.2f}\n".format(uniq_id, class_name, probablity)
        file.write(csv_str)
        file.write('END')



def test_cafa(data_test, learn_cls, df_test):
    learn_cls.data.add_test(data_test.train_ds)
    preds, _ = learn_cls.get_preds(ds_type=DatasetType.Test, ordered=True)

    write_result(df_test.uniq_id, preds, conf['local_project_path'] +
                 'result/' + 'result-' + conf['datetime'] + '.txt', learn_cls.data.classes)
