import numpy as np


def get_one_hot_encoded_array(label,label_map):
    for existing in label_map:
        if existing['name'] == label:
            return np.array(existing['value'])

def get_one_hot_encoded_array_for_label(label,label_dict,label_map):
    label = label_dict[label]
    for existing in label_map:
        if existing['name'] == label:
            return np.array(existing['value'])

def get_label_map(label_dict):
    unique_labels = list(label_dict.values()) 
    label_map = []
    for _,value in label_dict.items() :
        output_values = np.zeros(len(unique_labels), dtype=np.int)
        output_values [unique_labels.index(value)] = 1
        label_map.append({'name': value , 'value': output_values })
    return label_map


if __name__ == "__main__":
    label_dict = {'0': 'anger', '1': 'joy', '2': 'optimism', '3': 'sadness'}
    label_map = get_label_map(label_dict)

    print(get_one_hot_encoded_array_for_label("3",label_dict,label_map))