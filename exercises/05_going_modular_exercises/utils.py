import torch
from pathlib import Path


def save_checkpoint(model: torch.nn.Module,
                    target_dir: str,
                    model_name: str,
                    input_shape: int,
                    hidden_units: int,
                    output_shape: int,
                    class_names: list):

    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    assert model_name.endswith(".pth") or model_name.endswith(".pt"), \
        "Model name should end with '.pth' or '.pt'"

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "input_shape": input_shape,
        "hidden_units": hidden_units,
        "output_shape": output_shape,
        "class_names": class_names
    }

    save_path = target_dir_path / model_name

    print(f"[INFO] Saving checkpoint to: {save_path}")

    torch.save(checkpoint, save_path)


def load_checkpoint(model_path: str,
                    model_class,
                    device: torch.device):

    checkpoint = torch.load(model_path, map_location=device)

    model = model_class(
        input_shape=checkpoint["input_shape"],
        hidden_units=checkpoint["hidden_units"],
        output_shape=checkpoint["output_shape"]
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"[INFO] Loaded checkpoint from: {model_path}")

    return model, checkpoint["class_names"]