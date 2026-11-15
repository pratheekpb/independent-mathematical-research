
import sys
from pathlib import Path
import torch
import torch.nn as nn

sys.path.insert(0, str(Path("src").resolve()))

from data import get_dataloaders
from model import build_model

device = "cpu"

folder = Path("checkpoints_h18")
out = Path("outputs_h18")
out.mkdir(exist_ok=True)

# Load only epochs 1 and 10
c1 = torch.load(folder / "epoch_0001.pt", map_location=device, weights_only=False)
c10 = torch.load(folder / "epoch_0010.pt", map_location=device, weights_only=False)

s1 = c1["state_dict"]
s10 = c10["state_dict"]

# AD1 prediction
pred = {}

for key in s1:
    a, b = s1[key], s10[key]

    if not torch.is_floating_point(a):
        pred[key] = b.clone()

    elif "running_var" in key:
        # Positive variance: extrapolate in logarithmic space
        eps = 1e-8
        log_a = torch.log(a.clamp_min(eps))
        log_b = torch.log(b.clamp_min(eps))
        pred[key] = torch.exp(log_b + (8.0 / 9.0) * (log_b - log_a))

    else:
        pred[key] = b + (8.0 / 9.0) * (b - a)

# Save before revealing epoch 18
torch.save(pred, out / "ad1_prediction_epoch18_logvar.pt")
print("Prediction saved.")

# Reveal actual target for evaluation
actual_ckpt = torch.load(
    folder / "epoch_0018.pt",
    map_location=device,
    weights_only=False
)
actual = actual_ckpt["state_dict"]

# Parameter comparison: trainable parameters only
keys = [
    "net.0.weight", "net.0.bias",
    "net.1.weight", "net.1.bias",
    "net.4.weight", "net.4.bias",
    "net.5.weight", "net.5.bias",
    "net.8.weight", "net.8.bias"
]

error = torch.cat([
    (pred[k] - actual[k]).reshape(-1) for k in keys
])
actual_values = torch.cat([
    actual[k].reshape(-1) for k in keys
])

print("\nPARAMETER COMPARISON")
print("MAE:", error.abs().mean().item())
print("RMSE:", error.square().mean().sqrt().item())
print("Relative L2 error:", (
    error.norm() / actual_values.norm().clamp_min(1e-12)
).item())

# Recreate validation data
_, val_loader, _, _ = get_dataloaders(
    n_samples=1000,
    noise=0.20,
    batch_size=64,
    random_state=42
)

# Build and load models
pred_model = build_model("medium").to(device)
actual_model = build_model("medium").to(device)

pred_model.load_state_dict(pred)
actual_model.load_state_dict(actual)

pred_model.eval()
actual_model.eval()

criterion = nn.CrossEntropyLoss()

def evaluate(model):
    loss_sum = 0.0
    correct = 0
    total = 0
    outputs = []

    with torch.no_grad():
        for x, y in val_loader:
            logits = model(x)
            loss_sum += criterion(logits, y).item() * len(y)
            labels = logits.argmax(dim=1)
            correct += (labels == y).sum().item()
            total += len(y)
            outputs.append(labels)

    return loss_sum / total, correct / total, torch.cat(outputs)

pl, pa, pp = evaluate(pred_model)
al, aa, ap = evaluate(actual_model)

print("\nFUNCTIONAL COMPARISON")
print(f"Predicted loss: {pl:.6f}")
print(f"Actual loss: {al:.6f}")
print(f"Predicted accuracy: {pa * 100:.2f}%")
print(f"Actual accuracy: {aa * 100:.2f}%")
print(f"Prediction agreement: {(pp == ap).float().mean().item() * 100:.2f}%")
