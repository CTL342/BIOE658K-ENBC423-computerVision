import os
import glob
import csv
import torch

# Import model definition from train_all_models.py
try:
    from train_all_models import MBV2_CA as ModelClass
except ImportError:
    from train_all_models import MobileNetV2Custom as ModelClass

EXPERIMENT_DIRS = {
    "Baseline": "./experiments/baseline",
    "SE (r=24)": "./experiments/se",
    "CBAM (r=24)": "./experiments/cbam",
    "CA (r=32)": "./experiments/ca"
}

ARCH_SPECS = {
    "Baseline": {"attention": None, "reduction": 32},
    "SE (r=24)": {"attention": "se", "reduction": 24},
    "CBAM (r=24)": {"attention": "cbam", "reduction": 24},
    "CA (r=32)": {"attention": "ca", "reduction": 32}
}

OUTPUT_CSV = "experiment_results.csv"

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    print(f"\n{'='*72}")
    header = f"{'Model Architecture':<20} | {'Top-1 Acc':<12} | {'At Epoch':<10} | {'Params (M)':<12} | {'Delta Params'}"
    print(header)
    print('-'*72)

    baseline_params = None
    csv_rows = []

    for name, exp_path in EXPERIMENT_DIRS.items():
        spec = ARCH_SPECS[name]
        model = ModelClass(
            num_classes=10,
            width_mult=1.0,
            attention_type=spec["attention"],
            reduction=spec["reduction"]
        )
        params = count_parameters(model)
        if name == "Baseline":
            baseline_params = params

        ckpt_files = glob.glob(os.path.join(exp_path, "*best*.pth"))
        if not ckpt_files:
            delta_str = f"+{(params - baseline_params)/1e6:.3f}M" if baseline_params is not None else "0.000M"
            print(f"{name:<20} | {'Not Found':<12} | {'N/A':<10} | {params/1e6:<12.2f} | {delta_str}")
            csv_rows.append({
                "Model Architecture": name,
                "Top-1 Accuracy (%)": "Not Found",
                "Best Epoch": "N/A",
                "Parameters (M)": f"{params/1e6:.2f}",
                "Delta Parameters": delta_str
            })
            continue

        ckpt = torch.load(ckpt_files[0], map_location="cpu")
        acc = ckpt.get("best_top1", ckpt.get("best_acc", 0.0))
        ep = ckpt.get("epoch", 0)
        delta = f"+{(params - baseline_params)/1e6:.3f}M" if name != "Baseline" else "+0.000M"

        print(f"{name:<20} | {acc:<10.2f}% | {ep:<10} | {params/1e6:<12.2f} | {delta}")
        csv_rows.append({
            "Model Architecture": name,
            "Top-1 Accuracy (%)": f"{acc:.2f}",
            "Best Epoch": ep,
            "Parameters (M)": f"{params/1e6:.2f}",
            "Delta Parameters": delta
        })

    print(f"{'='*72}\n")

    # Save to CSV
    fieldnames = ["Model Architecture", "Top-1 Accuracy (%)", "Best Epoch", "Parameters (M)", "Delta Parameters"]
    with open(OUTPUT_CSV, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"Results successfully saved to {OUTPUT_CSV}")

if __name__ == '__main__':
    main()