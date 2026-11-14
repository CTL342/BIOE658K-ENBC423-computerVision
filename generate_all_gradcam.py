import os
import glob
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

# Import model definition from train_all_models.py
try:
    from train_all_models import MBV2_CA as ModelClass
except ImportError:
    from train_all_models import MobileNetV2Custom as ModelClass

EXPERIMENT_DIRS = {
    "Baseline": "./experiments/baseline",
    "SE": "./experiments/se",
    "CBAM": "./experiments/cbam",
    "CA": "./experiments/ca"
}

ARCH_SPECS = {
    "Baseline": {"attention": None, "reduction": 32},
    "SE": {"attention": "se", "reduction": 24},
    "CBAM": {"attention": "cbam", "reduction": 24},
    "CA": {"attention": "ca", "reduction": 32}
}

DATA_DIR = os.path.expanduser("~/workspace/imgproj/imagenette2-320/val")
OUTPUT_DIR = "./gradcam_results"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, x, class_idx=None):
        self.model.zero_grad()
        output = self.model(x)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        score = output[0, class_idx]
        score.backward()

        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze()
        cam = F.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam.cpu().numpy(), class_idx

def overlay_heatmap(img_np, cam_np):
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_np), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
    overlay = 0.5 * img_np + 0.5 * heatmap
    return np.clip(overlay, 0.0, 1.0)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    models = {}

    for name, exp_path in EXPERIMENT_DIRS.items():
        spec = ARCH_SPECS[name]
        model = ModelClass(
            num_classes=10,
            width_mult=1.0,
            attention_type=spec["attention"],
            reduction=spec["reduction"]
        )
        ckpt_files = glob.glob(os.path.join(exp_path, "*best*.pth"))
        if not ckpt_files:
            print(f"Warning: Checkpoint for {name} not found. Skipping.")
            continue
        
        ckpt = torch.load(ckpt_files[0], map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
        # Strip DDP prefix if present
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        model.to(DEVICE).eval()
        
        # Hook the final Inverted Residual block
        target_layer = model.features[-1]
        models[name] = GradCAM(model, target_layer)

    if not models:
        print("No models available for Grad-CAM.")
        return

    # Select representative samples from val set
    img_paths = sorted(glob.glob(os.path.join(DATA_DIR, "*", "*.JPEG")))[:4]
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    for idx, path in enumerate(img_paths, 1):
        raw_pil = Image.open(path).convert('RGB')
        raw_cropped = transforms.CenterCrop(224)(transforms.Resize(256)(raw_pil))
        img_np = np.array(raw_cropped) / 255.0
        input_tensor = preprocess(raw_pil).unsqueeze(0).to(DEVICE)

        fig, axes = plt.subplots(1, 1 + len(models), figsize=(16, 3.5))
        axes[0].imshow(img_np)
        axes[0].set_title("Input Image", fontsize=10, fontweight='bold', pad=10)
        axes[0].axis('off')

        for ax_idx, (m_name, cam_engine) in enumerate(models.items(), 1):
            cam, pred_cls = cam_engine(input_tensor)
            cam_resized = cv2.resize(cam, (224, 224))
            overlay = overlay_heatmap(img_np, cam_resized)
            axes[ax_idx].imshow(overlay)
            axes[ax_idx].set_title(f"{m_name} (Pred: {pred_cls})", fontsize=10, pad=10)
            axes[ax_idx].axis('off')

        out_path = os.path.join(OUTPUT_DIR, f"cam_comparison_{idx}.jpg")
        plt.subplots_adjust(top=0.82, bottom=0.05, left=0.02, right=0.98, wspace=0.08)
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {out_path}")

if __name__ == '__main__':
    main()