import json
from pathlib import Path

from fast_bert.prediction import BertClassificationPredictor

from definitions import ROOT_DIR

if __name__ == '__main__':
    model_dir = ROOT_DIR + '/model_output/model_out'
    label_dir = ROOT_DIR + '/data'
    predictor = BertClassificationPredictor(
        model_path=model_dir,
        label_path=label_dir,  # location for labels.csv file
        multi_label=False,
        model_type='bert',
        do_lower_case=False)

    test_filename = Path(ROOT_DIR) / 'data' / 'train.csv'
    test_data_list = []
    test_label_list = []
    with open(test_filename, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.replace('\n', '')
        item_list = line.split(',')
        annotation_type = item_list[-1]
        sentence = ','.join(item_list[1:-1])
        test_data_list.append(sentence)
        test_label_list.append(annotation_type)
    multiple_predictions = predictor.predict_batch(test_data_list)
    print(multiple_predictions)

    with open('predictions.json', 'w') as f:
        json.dump(multiple_predictions, f)
