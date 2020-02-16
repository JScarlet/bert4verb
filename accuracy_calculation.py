import json

if __name__ == '__main__':
    with open('data/predictions.json', 'r') as f:
        predictions = json.load(f)

    for each in predictions:
        print(each)