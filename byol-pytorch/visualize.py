import os
# load safetensor file
import torch
from safetensors.torch import load_file
import open3d as o3d
from huggingface_hub import hf_hub_download, list_repo_files

def load_safetensor_model(data_path):  
    data = load_file(data_path)
    return data
def show_image_from_numpy(image_numpy):
    import matplotlib.pyplot as plt
    plt.imshow(image_numpy)
    plt.axis('off')
    plt.show()


def load_safetensor_from_hf(repo_id, filename, repo_type="dataset"):
    cached_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision="7bb7c7f3d379c5145bb06d2cf0949c66ac9a2c4e",
        repo_type=repo_type,
        local_files_only=True
    )
    return load_file(cached_path)


data = load_safetensor_from_hf('MatchLab/PointMapVerse', 'light_3rscan/02b33df9-be2b-2d54-9062-1253be3ce186.safetensors')
images = data['color_images']


for img in images:
    show_image_from_numpy(img)






