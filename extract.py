import pandas as pd
import numpy as np
import json

def extract_mapping(map_filename):
    mapping = {}
    with open(map_filename,"r") as f:
        for line in f.readlines():
            data = line.strip().split()
            mapping[data[0]] = data[1]
    return mapping


def load_label_map(filename):
    df = pd.read_csv(filename)

    emotions = df['sentiment'].tolist()
    del df

    unique_labels = list(set(emotions))

    label_map = []

    for label in unique_labels:
        output_values = np.zeros(len(unique_labels), dtype=np.int)
        output_values [unique_labels.index(label)] = 1
        label_map.append({'name': label , 'value': output_values })
    
    return label_map

def load_data_set(filename):
    df = pd.read_csv(filename)
    content = df['content'].tolist()
    labels = df['sentiment'].tolist()
    return content, labels



def extract_content_labels(filename):
    contents = []
    labels = []

    content_file = filename+"_text.txt"
    label_file = filename+"_labels.txt"

    with open(content_file,"r") as f:
        for line in f.readlines():
            line = line.replace("\n","").strip()
            contents.append(line)

    with open(label_file,"r") as f:
        for line in f.readlines():
            line = line.replace("\n","").strip()
            labels.append(line)

    return contents,labels

def extract_hyper_param(filename):
    with open(filename,"r") as f:
        data = json.loads(f.read())
    return data

if __name__ == "__main__":
    mapping = extract_mapping("./data/mapping.txt")    
    print(mapping)

    a ,b = extract_content_labels("./data/train")
    assert len(a) == len(b)
    a ,b = extract_content_labels("./data/test")
    assert len(a) == len(b)
    a ,b = extract_content_labels("./data/val")
    assert len(a) == len(b)

    print(extract_hyper_param("./model/hyper_param.json"))


    