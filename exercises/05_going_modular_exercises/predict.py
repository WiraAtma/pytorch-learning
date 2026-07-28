import torch
import utils
import model_builder
import engine

from torchvision import transforms
from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "cpu"

# path gambar yang ingin di prediksi
image_path = Path("data/prediction_gambar.jpg")

data_transform = transforms.Compose([
    transforms.Resize((64, 64))
])

model, class_names = utils.load_checkpoint(
    model_path="models/05_going_modular_script_mode_tinyvgg_model_exercises.pth",
    model_class=model_builder.TinyVgg,
    device=device
)

engine.prediction(
  model=model,
  image_path=image_path,
  class_names=class_names,
  transform=data_transform,
  device=device
)