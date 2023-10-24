from cuml.manifold.umap import UMAP
import numpy as np
import os
import gc
import torch

gc.collect()
torch.cuda.empty_cache()

def read_npy(data_path):
    data = np.load(data_path)
    print("Data loaded.", data.shape)
    return data


def calculate_umap(data, n_neighbors=15):
    parameters = {"min_dist": 0.1, "spread": 1.0}
    print("UMAP calculation has started.")
    result = UMAP(
        n_components=6,
        n_neighbors=n_neighbors,
        min_dist=parameters["min_dist"],
        spread=parameters["spread"],
    ).fit_transform(data)
    return result


def save_embeddings(data_path, umap_embeddings):
    output_folder = "umap"
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(
        output_folder, os.path.splitext(os.path.basename(data_path))[0] + "_umap.npy"
    )
    np.save(output_path, umap_embeddings)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    data_paths = [
        "/mnt/c/Users/Mert/Desktop/florfaunea/pca/sample_fauna_embeddings_cleaned_pca.npy",
        "/mnt/c/Users/Mert/Desktop/florfaunea/pca/sample_flora_embeddings_cleaned_pca.npy",
        "/mnt/c/Users/Mert/Desktop/florfaunea/pca/sample_fungi_embeddings_cleaned_pca.npy",
    ]
    for data_path in data_paths:
        data = read_npy(data_path)
        umap_embeddings = calculate_umap(data)
        save_embeddings(data_path, umap_embeddings)
