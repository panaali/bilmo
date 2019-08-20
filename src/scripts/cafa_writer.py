import pandas as pd

def write_result(uniq_ids, preds, go_ids, path, prediction_classes):
    header = ('AUTHOR ARODZ\n' +
              'MODEL 1\n' +
              'KEYWORDS Neural Networks, Language Modeling\n')
    with open(path, 'a') as file:
        file.write(header)
        for i, go_id in enumerate(prediction_classes):
            df = pd.DataFrame(
                {'cafa_id': uniq_ids, 'go_id': go_id, 'preds': preds[:, i]})
            file.write(df.to_csv(index=False, header=False, sep='\t'))
