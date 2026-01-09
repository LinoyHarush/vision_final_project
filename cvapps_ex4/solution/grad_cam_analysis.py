"""Show network train graphs and analyze training results."""
import os
import argparse

import torch
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM

from torch.utils.data import DataLoader

from common import FIGURES_DIR
from utils import load_dataset, load_model


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Arguments
def parse_args():
    """Parse script arguments.

    Returns:
        Namespace with model name, checkpoint path and dataset name.
    """
    parser = argparse.ArgumentParser(description='Analyze network performance.')
    parser.add_argument('--model', '-m',
                        default='XceptionBased', type=str,
                        help='Model name: SimpleNet or XceptionBased.')
    parser.add_argument('--checkpoint_path', '-cpp',
                        default='checkpoints/XceptionBased.pt', type=str,
                        help='Path to model checkpoint.')
    parser.add_argument('--dataset', '-d',
                        default='fakes_dataset', type=str,
                        help='Dataset: fakes_dataset or synthetic_dataset.')

    return parser.parse_args()


def get_grad_cam_visualization(test_dataset: torch.utils.data.Dataset,
                               model: torch.nn.Module) -> tuple[np.ndarray,
                                                                torch.tensor]:
    """Return a tuple with the GradCAM visualization and true class label.

    Args:
        test_dataset: test dataset to choose a sample from.
        model: the model we want to understand.

    Returns:
        (visualization, true_label): a tuple containing the visualization of
        the conv3's response on one of the sample (256x256x3 np.ndarray) and
        the true label of that sample (since it is an output of a DataLoader
        of batch size 1, it's a tensor of shape (1,)).
    """
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    # Ensure we use the "real" model if wrapped (DataParallel/DistributedDataParallel)
    core_model = model.module if hasattr(model, "module") else model
    device = next(core_model.parameters()).device
    core_model.eval()

    # b) Sample a single image (batch_size=1, shuffle=True)
    loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    x, y = next(iter(loader))  # x: (1,C,H,W), y: (1,)
    x = x.to(device)

    # c) Compute Grad-CAM for target layer: model.conv3
    target_layer = core_model.conv3

    # Choose target class:
    # Prefer predicted class if multi-logit output, else let GradCAM handle scalar output.
    with torch.enable_grad():
        out = core_model(x)

    if out.ndim == 2 and out.shape[1] >= 2:
        class_idx = int(out.argmax(dim=1).item())
        targets = [ClassifierOutputTarget(class_idx)]
    else:
        targets = None  # single-logit / scalar output case

    # Grad-CAM needs gradients and must hook layers from the same model object
    with torch.enable_grad():
        cam = GradCAM(model=core_model, target_layers=[target_layer])
        grayscale_cam = cam(input_tensor=x, targets=targets)[0]  # (H, W)

    # d) Create visualization overlay (HxWx3)
    img = x[0].detach().float().cpu()  # (C,H,W)
    if img.shape[0] == 1:
        img = img.repeat(3, 1, 1)
    rgb = img.permute(1, 2, 0).numpy()  # (H,W,3)

    # Robust scaling to [0,1] even if input was normalized
    rgb = rgb - np.min(rgb)
    mx = np.max(rgb)
    if mx > 1e-8:
        rgb = rgb / mx
    rgb = np.clip(rgb, 0.0, 1.0)

    visualization = show_cam_on_image(rgb, grayscale_cam, use_rgb=True)  # uint8 (H,W,3)

    return visualization, y


def main():
    """Create two GradCAM images, one of a real image and one for a fake
    image for the model and dataset it receives as script arguments."""
    args = parse_args()
    test_dataset = load_dataset(dataset_name=args.dataset, dataset_part='test')

    model_name = args.model
    model = load_model(model_name)
    model.load_state_dict(torch.load(args.checkpoint_path)['model'])

    model.eval()
    seen_labels = []
    while len(set(seen_labels)) != 2:
        visualization, true_label = get_grad_cam_visualization(test_dataset,
                                                               model)
        grad_cam_figure = plt.figure()
        plt.imshow(visualization)
        title = 'Fake Image' if true_label == 1 else 'Real Image'
        plt.title(title)
        seen_labels.append(true_label.item())
        grad_cam_figure.savefig(
            os.path.join(FIGURES_DIR,
                         f'{args.dataset}_{args.model}_'
                         f'{title.replace(" ", "_")}_grad_cam.png'))


if __name__ == "__main__":
    main()
