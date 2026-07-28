#%% writefile 05_going_modular_exercises/get_data.py

import os 
import requests
import zipfile
from pathlib import Path

data_path = Path("/data")
image_path = data_path / "pizza_steak_sushi_20_percent"

# os.remove(image_path)

if image_path.is_dir():
  print(f"{image_path} folder sudah tersedia")
else:
  print(f"Folder {image_path} tidak tersedia, membuat directory baru...")
  image_path.mkdir(parents=True, exist_ok=True)

with open(data_path / "pizza_steak_sushi_20_percent.zip", "wb") as f:
  request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi_20_percent.zip")
  print("Mendownload Data....")
  f.write(request.content)

with zipfile.ZipFile(data_path / "pizza_steak_sushi_20_percent.zip", "r") as zip_ref:
  print("Unzipping File...")
  zip_ref.extractall(image_path)

print("Download Data Berhasil!")

os.remove(data_path / "pizza_steak_sushi_20_percent.zip")