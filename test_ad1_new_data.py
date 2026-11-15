
import sys
from pathlib import Path
import torch
import torch.nn as nn

sys.path.insert(0, str(Path("src").resolve()))

from data import get_dataloaders
from model import build_model

device = "cpu"

# 1. Generate an independent dataset.
# Different seed ensures new examples.
_, test_loader, _, _ = get_dataloaders(
    n_samples=1000,
    noise=0.20,
    batch_size=64,
    random_state=12345
)

# Its held-out split contains 200 examples.

# 2. Load predicted and actual models
pred_state = torch.load(
    "outputs_h18/ad1_prediction_epoch18_logvar.pt",
    map_location=device,
    weights_only=False
)

actual_ckpt = torch.load(
    "checkpoints_h18/epoch_0018.pt",
    map_location=device,
    weights_only=False
)

actual_state = actual_ckpt["state_dict"]

pred_model = build_model("medium").to(device)
actual_model = build_model("medium").to(device)

pred_model.load_state_dict(pred_state)
actual_model.load_state_dict(actual_state)

pred_model.eval()
actual_model.eval()

criterion = nn.CrossEntropyLoss()

# 3. Evaluate both on the same 200 examples
def evaluate(model):
    total_loss = 0.0
    correct = 0
    total = 0
    predictions = []

    with torch.no_grad():
        for X, y in test_loader:
            logits = model(X)
            loss = criterion(logits, y)
            labels = logits.argmax(dim=1)

            total_loss += loss.item() * len(y)
            correct += (labels == y).sum().item()
            total += len(y)
            predictions.append(labels)

    return (
        total_loss / total,
        correct,
        total,
        torch.cat(predictions)
    )

p_loss, p_correct, total, p_labels = evaluate(pred_model)
a_loss, a_correct, _, a_labels = evaluate(actual_model)

agreement = (p_labels == a_labels).sum().item()

# 4. Results
print("\n===== INDEPENDENT AD1 TEST =====")
print(f"New examples: {total}")

print("\nH-PREDICTED MODEL")
print(f"Correct: {p_correct}/{total}")
print(f"Accuracy: {100*p_correct/total:.2f}%")
print(f"Loss: {p_loss:.6f}")

print("\nACTUAL EPOCH-18 MODEL")
print(f"Correct: {a_correct}/{total}")
print(f"Accuracy: {100*a_correct/total:.2f}%")
print(f"Loss: {a_loss:.6f}")

print("\nMODEL AGREEMENT")
print(f"Same predictions: {agreement}/{total}")
print(f"Agreement: {100*agreement/total:.2f}%")
print("================================")
