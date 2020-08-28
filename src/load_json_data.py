import json
import numpy as np 
import os

def load_json_data(json_path):
    with open(json_path, 'rb') as f:
        data = json.load(f)
    features = np.array(data[0]).astype(float)
    targets = 2 * (np.array(data[1]).astype(float) - 1) - 1

    return features, targets

if __name__ == "__main__":
    try:
        import matplotlib.pyplot as plt
    except:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    
    data_files = [
        os.path.join('../data', x) 
        for x in os.listdir('../data/') 
        if x[-4:] == 'json']

    for data_file in data_files:
        features, targets = load_json_data(data_file)
        plt.figure(figsize=(6,4))
        plt.scatter(features[:, 0], features[:, 1], c=targets)
        plt.title(data_file)
        plt.savefig(f'../data/{data_file}.png')
        print(features.shape, targets.shape, data_file)