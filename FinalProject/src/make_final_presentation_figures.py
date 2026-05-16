import os
import matplotlib.pyplot as plt

out_dir = "figures"
os.makedirs(out_dir, exist_ok=True)

labels = ["FP32", "INT8", "Compiled"]

latency = [435.28, 238.40, 394.05]
throughput = [18.38, 33.56, 20.30]
model_size = [255.45, 132.29, 255.45]
correct = [789, 782, 789]

def bar_chart(values, title, ylabel, filename, ylim=None):
    plt.figure(figsize=(8, 5))
    plt.bar(labels, values)
    plt.title(title)
    plt.ylabel(ylabel)
    if ylim is not None:
        plt.ylim(*ylim)
    for i in range(len(values)):
        text = f"{values[i]:.2f}" if isinstance(values[i], float) else str(values[i])
        plt.text(i, values[i], text, ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, filename), dpi=200)
    plt.close()

bar_chart(
    latency,
    "Batch-8 Latency Comparison",
    "Latency per Batch (ms)",
    "presentation_batch8_latency.png",
    ylim=(0, 520)
)

bar_chart(
    throughput,
    "Batch-8 Throughput Comparison",
    "Throughput (samples/sec)",
    "presentation_batch8_throughput.png",
    ylim=(0, 40)
)

bar_chart(
    model_size,
    "Model Size Comparison",
    "Model Size (MB)",
    "presentation_model_size.png",
    ylim=(0, 280)
)

bar_chart(
    correct,
    "Batch-8 Accuracy as Correct Predictions",
    "Correct Predictions out of 872",
    "presentation_correct_predictions.png",
    ylim=(0, 872)
)

print("Regenerated final presentation figures in:", out_dir)
