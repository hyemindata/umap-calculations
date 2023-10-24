import numpy as np
from sklearn.decomposition import PCA
import os
from tqdm import tqdm

def read_npy(data_path):
    data = np.load(data_path)
    filelist = data["filelist"]
    embeddings = data["embeddings"]
    print("Data loaded.", embeddings.shape)
    return filelist, embeddings


def calculate_pca(data):
    print("Calculating PCA...")
    pca = PCA(n_components=224)
    pca_embeddings = pca.fit_transform(data)
    return pca_embeddings


def save_embeddings(data_path, pca_embeddings):
    output_folder = "pca"
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(
        output_folder, os.path.splitext(os.path.basename(data_path))[0] + "_pca.npy"
    )
    np.save(output_path, pca_embeddings, allow_pickle=True)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    data_paths = [
        "embeddings\sample_fauna_embeddings_cleaned.npz",
        "embeddings\sample_flora_embeddings_cleaned.npz",
        "embeddings\sample_fungi_embeddings_cleaned.npz",
    ]

    for data_path in tqdm(data_paths):
        _, embeddings = read_npy(data_path)
        pca_embeddings = calculate_pca(embeddings)
        save_embeddings(data_path, pca_embeddings)
