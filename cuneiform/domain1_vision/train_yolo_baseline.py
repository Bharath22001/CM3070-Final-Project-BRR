import warnings
warnings.filterwarnings("ignore")

import torch
import torch_directml
import ultralytics.utils.torch_utils as torch_utils
import ultralytics.engine.trainer as trainer_module
from ultralytics import YOLO

# DirectML patch 1: torch.cdist is unstable on DirectML, so I fall back to a
# manual L1 distance on that device.
original_cdist = torch.cdist
def dml_safe_cdist(x1, x2, *args, **kwargs):
    if x1.device.type == 'privateuseone':
        p = kwargs.get('p', args[0] if args else 2.0)
        if p == 1:
            return torch.abs(x1.unsqueeze(1) - x2.unsqueeze(0)).sum(dim=-1)
    return original_cdist(x1, x2, *args, **kwargs)
torch.cdist = dml_safe_cdist

# DirectML patch 2: Ultralytics' select_device() doesn't know DirectML, so I
# pass the device straight through.
dml_device = torch_directml.device()
original_select_device = torch_utils.select_device

def dml_select_device(device='', *args, **kwargs):
    if isinstance(device, torch.device) and device.type == 'privateuseone':
        return device
    if isinstance(device, str) and 'privateuseone' in device:
        return dml_device
    return original_select_device(device, *args, **kwargs)

torch_utils.select_device = dml_select_device
trainer_module.select_device = dml_select_device  # patch the imported reference too

# DirectML patch 3: DirectML has no kernel for torch.unique(return_counts=True),
# which YOLOv8's loss uses to count labels, so I run that one op on CPU.
original_unique = torch.unique
def dml_safe_unique(input, *args, **kwargs):
    if torch.is_tensor(input) and input.device.type == 'privateuseone':
        result = original_unique(input.cpu(), *args, **kwargs)
        if isinstance(result, tuple):
            return tuple(t.to(input.device) if torch.is_tensor(t) else t for t in result)
        return result.to(input.device)
    return original_unique(input, *args, **kwargs)
torch.unique = dml_safe_unique


def main():
    print("Initializing AMD DirectML device...")
    print(f"Routing YOLOv8 training to: {dml_device}")

    print("Loading YOLOv8s architecture...")
    model = YOLO('yolov8s.pt')

    print("\nStarting Baseline Training...")
    results = model.train(
        data='./yolo_dataset/data.yaml',
        epochs=100,
        imgsz=512,
        batch=8,
        device=dml_device,
        project='cuneiform_yolo',
        name='baseline_run',
        save=True,
        plots=True,
        amp=False,
        workers=0,
    )

    print("\n--- YOLOv8 TRAINING COMPLETE ---")
    print("Check the 'cuneiform_yolo/baseline_run' folder for weights and evaluation graphs.")


if __name__ == '__main__':
    main()