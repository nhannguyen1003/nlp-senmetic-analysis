import json
import csv

import pandas as pd

from tqdm import tqdm

def read_result(path):
    result = json.load(open(path, 'r'))
    return result
    

def read_original_data(path):
    data = pd.read_csv(path)
    lines = []
    for idx, sentence in enumerate(data['Data']):
        lines.append([idx, data['Class'][idx], sentence])
    
    return lines

def write_result(result, path_save):
    header = ["Class", "Data"]
    with open(path_save, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)

        # write the header
        writer.writerow(header)

        # write the data
        for _, line in enumerate(result):
            writer.writerow([*line])
            
    
def main():
    augment_results = read_result("./augment_data/result/new_data.json")
    original_data = read_original_data("./dataset/original/vlsp_sentiment_train.csv")
    results = []
    
    for line in tqdm(original_data):
        results.append(line)
        if str(line[0]) in augment_results:
            for new_sentence in augment_results[str(line[0])]:
                if new_sentence != "":
                    results.append([line[0], line[1], new_sentence])
    
    write_result(result=results, path_save="./dataset/augment_gpt/vlsp_sentiment_train.csv")
    
    
if __name__ == '__main__':
    main()