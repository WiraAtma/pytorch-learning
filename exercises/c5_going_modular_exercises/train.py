import torch
import data_setup, engine, model_builder, utils
from torchvision import transforms

NUM_EPOCHS = 20
BATCH_SIZE = 64
HIDDEN_UNITS = 10
LEARNING_RATE = 0.003

train_dir = "data/pizza_steak_sushi_20_percent/train"
test_dir = "data/pizza_steak_sushi_20_percent/test"

device = "cuda" if torch.cuda.is_available() else "cpu"

def main():
  data_tranform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.TrivialAugmentWide(),
    transforms.ToTensor()
  ])

  train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir=train_dir,
                                                                                 test_dir=test_dir,
                                                                                 transform=data_tranform,
                                                                                 batch_size=BATCH_SIZE)

  model = model_builder.TinyVgg(input_shape=3,
                                hidden_units=HIDDEN_UNITS,
                                output_shape=len(class_names)).to(device)

  loss_fn = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(params=model.parameters(),
                               lr=LEARNING_RATE)

  engine.train(model=model,
               train_dataloader=train_dataloader,
               test_dataloader=test_dataloader,
               loss_fn=loss_fn,
               optimizer=optimizer,
               epochs=NUM_EPOCHS,
               device=device)

  utils.save_checkpoint(model=model,
                        target_dir="models",
                        model_name="05_going_modular_script_mode_tinyvgg_model_exercises.pth",
                        input_shape=3,
                        hidden_units=10,
                        output_shape=len(class_names),
                        class_names=class_names)

if __name__ == "__main__":
  main()