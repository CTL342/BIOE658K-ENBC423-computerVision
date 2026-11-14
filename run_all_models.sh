#!/bin/bash
#SBATCH --job-name=mbv2_ca_reproduction
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --gpus=3
#SBATCH --cpus-per-task=8
#SBATCH --time=18:00:00
#SBATCH --output=reproduction_%j.log
#SBATCH --error=reproduction_%j.err

set -e

# 1. Activate the environment
source ~/venv/bin/activate

DATA_DIR="$HOME/workspace/imgproj/imagenette2-320"
if [ ! -d "$DATA_DIR/train" ] || [ ! -d "$DATA_DIR/val" ]; then
    echo "ERROR: ImageNette dataset not found at $DATA_DIR" >&2
    exit 1
fi

echo "=========================================================="
echo "Starting Official CVPR 2021 Coordinate Attention Training"
echo "Host: $(hostname)"
echo "GPUs: 3x RTX 3090 (DistributedDataParallel)"
echo "Dataset: $DATA_DIR"
echo "=========================================================="

# Model 1: Baseline MobileNetV2 (No attention)
echo "--> [1/4] Training Baseline MobileNetV2..."
srun python -u train_all_models.py \
    --data-dir "$DATA_DIR" \
    --attention none \
    --epochs 200 \
    --batch-size 64 \
    --lr 0.05 \
    --output-dir experiments/baseline

# Model 2: MobileNetV2 + SE (r=24, Hu et al.)
echo "--> [2/4] Training MobileNetV2 + Squeeze-and-Excitation (r=24)..."
srun python -u train_all_models.py \
    --data-dir "$DATA_DIR" \
    --attention se \
    --reduction 24 \
    --epochs 200 \
    --batch-size 64 \
    --lr 0.05 \
    --output-dir experiments/se

# Model 3: MobileNetV2 + CBAM (r=24, Woo et al.)
echo "--> [3/4] Training MobileNetV2 + CBAM (r=24)..."
srun python -u train_all_models.py \
    --data-dir "$DATA_DIR" \
    --attention cbam \
    --reduction 24 \
    --epochs 200 \
    --batch-size 64 \
    --lr 0.05 \
    --output-dir experiments/cbam

# Model 4: MobileNetV2 + Coordinate Attention (r=32, Hou et al. 2021)
echo "--> [4/4] Training MobileNetV2 + Coordinate Attention (r=32)..."
srun python -u train_all_models.py \
    --data-dir "$DATA_DIR" \
    --attention ca \
    --reduction 32 \
    --epochs 200 \
    --batch-size 64 \
    --lr 0.05 \
    --output-dir experiments/ca

echo "=========================================================="
echo "All 4 models successfully trained!"
echo "Evaluating checkpoints..."
python evaluate_all_results.py
echo "Generating Grad-CAM visualizations..."
python generate_all_gradcam.py
echo "=========================================================="