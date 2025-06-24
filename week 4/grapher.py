import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from matplotlib.lines import Line2D
import itertools

# Base directory
base_dir = Path("caltech101_log")

# Get all folders that contain training logs, sorted alphabetically
folders = sorted([f for f in base_dir.iterdir() if f.is_dir() and (f / "train_loss.csv").exists()],
                 key=lambda f: f.name.lower())

# Color cycling
color_cycle = itertools.cycle(plt.cm.tab10.colors)
model_colors = {}

# Prepare figure and axes
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
ax1.set_title("Train vs Test Loss")
ax1.set_xlabel("Images Seen")
ax1.set_ylabel("Loss")

ax2.set_title("Train vs Test Accuracy")
ax2.set_xlabel("Images Seen")
ax2.set_ylabel("Accuracy")

# For custom legend
custom_lines = []

for folder in folders:
    model_name = folder.name
    color = model_colors[model_name] = next(color_cycle)

    try:
        train_loss = pd.read_csv(folder / "train_loss.csv")
        test_loss = pd.read_csv(folder / "test_loss.csv")
        train_acc = pd.read_csv(folder / "train_acc.csv")
        test_acc = pd.read_csv(folder / "test_acc.csv")
    except Exception as e:
        print(f"⚠️ Skipping {model_name}: {e}")
        continue

    # Plot loss curves
    ax1.plot(train_loss["images_seen"], train_loss["value"], linestyle="--", color=color)
    ax1.plot(test_loss["images_seen"], test_loss["value"], linestyle="-", color=color)

    # Plot accuracy curves
    ax2.plot(train_acc["images_seen"], train_acc["value"], linestyle="--", color=color)
    ax2.plot(test_acc["images_seen"], test_acc["value"], linestyle="-", color=color)

    # Add to custom legend (solid line for consistency)
    custom_lines.append(Line2D([0], [0], color=color, lw=2, label=model_name, linestyle="-"))

# Add legends
ax1.legend(handles=custom_lines, title="Models (Loss)", loc="upper right")
ax2.legend(handles=custom_lines, title="Models (Accuracy)", loc="lower right")

plt.tight_layout()
plt.savefig("training_results.png")
plt.show()

print("✅ Plot saved as 'training_results.png'")
