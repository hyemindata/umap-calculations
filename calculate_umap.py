import os
import gc
import json
import argparse
from datetime import datetime
from pathlib import Path

import torch
import numpy as np
from cuml.manifold.umap import UMAP

gc.collect()
torch.cuda.empty_cache()

def read_npy(data_path):
    data = np.load(data_path)
    print("Data loaded.", data.shape)
    return data

def calculate_umap(data, umap_params):
    print("UMAP calculation has started.")
    print(umap_params)
    result = UMAP(**umap_params).fit_transform(data)
    return result

def save_embeddings(save_path, umap_embeddings):
    np.save(save_path, umap_embeddings)
    print(f"Saved to {save_path}")

def get_timestamp():
    timestamp = datetime.now().strftime("%d%m%Y-%H%M%S")
    return timestamp

def prepare_save_path(data_path, idx, save_dir, timestamp):
    title = Path(data_path).stem
    umap_title = f'{title}_{timestamp}'

    exp_title = f'{umap_title}_{idx:05d}'

    umap_save_dir = Path(save_dir) / umap_title / exp_title
    os.makedirs(umap_save_dir, exist_ok=True)

    umap_save_path = umap_save_dir / (exp_title+'.npy')
    config_save_path = umap_save_dir / (exp_title+'.json')
    return umap_save_path, config_save_path

def save_config(save_path, umap_params):
    with open(save_path, 'w') as f:
        json.dump(umap_params, f)
    
def main(args):
    timestamp = get_timestamp()
    for data_path in args.pca:
        data = read_npy(data_path)

        idx = 0
        for nn in args.n_neighbors:
            for md in args.min_dist:
                for sp in args.spread:
                    for mt in args.metric:
                        for ns in args.negative_sample_rate:
                            for rs in args.repulsion_strength:
                                umap_params = {"n_components" : 6, "n_neighbors" : nn,
                                               "min_dist" : md, "spread" : sp, "metric" :mt,
                                               "negative_sample_rate" : ns, "repulsion_strength" : rs}
                                umap_embeddings = calculate_umap(data, umap_params=umap_params)
                                umap_path, conf_path = prepare_save_path(data_path, idx, args.output, timestamp)
                                save_embeddings(umap_path, umap_embeddings)
                                save_config(conf_path, umap_params)
                                idx += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pca', type=str, nargs='*', help='Path to PCA npy')
    parser.add_argument('--n_neighbors', type=int, nargs='*', default=[12])
    parser.add_argument('--min_dist', type=float, nargs='*', default=[0.01])
    parser.add_argument('--spread', type=float, nargs='*', default=[1.0])
    parser.add_argument('--metric', type=str, nargs='*', default=["euclidean"])
    parser.add_argument('--negative_sample_rate', type=int, nargs='*', default=[5])
    parser.add_argument('--repulsion_strength', type=float, nargs='*', default=[1.0])

    parser.add_argument('--output', type=str, default='./output', help='Umap save directory')
    args = parser.parse_args()

    main(args)
