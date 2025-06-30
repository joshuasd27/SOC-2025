"""
Contains func to download Kaggle 7000 Pokemon dataset and moves it to ../data/raw/pokemon_images.
Run from scripts folder as working directory (get working directory by "import os \n os.get_cwd())
"""

import kagglehub
import shutil
from pathlib import Path

def download_dataset(kaggle_url:str, target_path:Path):
    """
    Download kaggle dataset into cache and move it into target_path

    Args:
        kaggle_url (str): Eg., username/dataset_name. Get by kaggle datasets list --search <search words>
        target_path (Path): Where to move contents/actual DataFolder into

    Returns:
    """
    # Download and extract dataset into ~/.cache
    downloaded_path_str = kagglehub.dataset_download(kaggle_url) #absolute path to actual Data folder parent, usually dwonloaded into ~/.cache
    downloaded_path = Path(downloaded_path_str) #if ls downoaded_path will get PokemonData
    shutil.copytree(downloaded_path, target_path, dirs_exist_ok=True) # makes PokemonData dir inside target_path


kaggle_url = "lantian773030/pokemonclassification"
target_path = Path("../data/raw")


if (target_path/"PokemonData/Abra").exists():
    print("PokemonData Image Folder already downloaded.")
    print("Example folder: ", (target_path/"PokemonData/Abra").absolute())
    print("\n")
else:
    download_dataset(kaggle_url=kaggle_url, target_path=target_path)
    if (target_path/"PokemonData/Abra").exists():
        print("PokemonData Image Folder succesfully downloaded.")
        print("Example image folder absolute path: ", (target_path/"PokemonData/Abra").absolute())
        print("\n")
