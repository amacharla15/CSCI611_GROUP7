# Efficient Transformer Inference for Text Classification

**Course:** CSCI 611  
**Project:** Final Project  
**Team:** Abinesh · Akshith · Arushi  
**Task:** SST-2 binary sentiment classification  
**Model:** DistilBERT (`distilbert-base-uncased`)  
**Optimization methods:** Dynamic INT8 quantization and `torch.compile`

---

## 1. Project Overview

This project studies efficient transformer inference for text classification. We fine-tuned a DistilBERT model on the full SST-2 training split and compared three inference variants:

1. **FP32 baseline**
2. **Dynamic INT8 quantized model**
3. **Compiled FP32 model using `torch.compile`**

The goal was to evaluate inference tradeoffs across:

- validation accuracy
- correct prediction count
- latency
- throughput
- model size
- compile overhead
- prediction changes after quantization

This project focuses on deployment-style analysis, not only model accuracy.

---

## 2. Dataset

We used the SST-2 sentiment classification task.

| Split | Examples | Purpose |
|---|---:|---|
| Train | 67,349 | Fine-tuning DistilBERT |
| Validation | 872 | Local evaluation and benchmarking |

Labels:

```text
0 = negative
1 = positive

We report validation accuracy, not test accuracy, because the SST-2 validation split is the labeled split used for local scoring.

3. Final Training Setup
Item	Value
Base model	distilbert-base-uncased
Task	Binary sequence classification
Training examples	67,349
Validation examples	872
Epochs	1
Seed	42
Training hardware	University A100 GPU
Final inference benchmark	Local CPU / WSL
Final comparison batch size	8

The A100 GPU was used for full-dataset fine-tuning. CPU inference benchmarking was used because PyTorch dynamic quantization mainly targets CPU Linear layer inference.

4. Final Batch-8 Results

These are the final reported results from the full-SST-2 trained checkpoint.

Variant	Accuracy	Correct / 872	Latency	Speedup	Throughput	Model Size	Compile First Call
FP32 baseline	0.9048	789	435.28 ms	1.00x	18.38/s	255.45 MB	0.00 s
Dynamic INT8	0.8968	782	238.40 ms	1.83x	33.56/s	132.29 MB	0.00 s
Compiled FP32	0.9048	789	394.05 ms	1.10x	20.30/s	255.45 MB	81.52 s

Main takeaway:

Dynamic INT8 quantization gave the strongest size and latency improvement.
INT8 reduced model size by about 48.2%.
INT8 improved batch-8 latency by about 1.83x.
INT8 caused a small accuracy drop from 90.48% to 89.68%.
torch.compile preserved accuracy and slightly improved steady-state latency, but had high first-call compile overhead.
5. What Each Optimization Does
FP32 Baseline

The FP32 baseline is the normal fine-tuned DistilBERT model before optimization. It uses normal PyTorch eager execution and FP32 weights.

Dynamic INT8 Quantization

Dynamic quantization was applied mainly to torch.nn.Linear layers.

A Linear layer performs:

output = input × weight + bias

DistilBERT contains many Linear layers in attention, feed-forward blocks, and the classifier head. Dynamic quantization stores Linear weights using INT8 values plus scale metadata. This reduces model size and can speed up CPU matrix multiplication.

torch.compile

torch.compile does not compress weights. It changes the execution path by capturing PyTorch operations into graph regions and compiling them for repeated execution.

For the full model, graph inspection showed:

Graph count: 1
Graph breaks: 0
Captured operations: 102

Profiler inspection showed compiled execution used a CompiledFxGraph / Torch-Compiled Region. Total self CPU time decreased from 4.606 s to 3.796 s, but aten::addmm, corresponding to Linear-layer matrix multiplication, remained the dominant bottleneck.

6. Repository Structure
FinalProject/
├── README.md
├── FINAL_REPORT.md
├── PROJECT_PHASE_LOG.md
├── requirements.txt
├── src/
│   ├── benchmark_all.py
│   ├── train_baseline.py
│   ├── train_baseline_seeded.py
│   ├── compare_fp32_int8_predictions.py
│   ├── capture_compile_graph.py
│   ├── profile_eager_vs_compile_full.py
│   ├── inspect_torch_compile.py
│   └── make_presentation_figures.py
├── results/
│   ├── final_results.csv
│   ├── final_summary.md
│   ├── local_full_seed42_batch8_compile_log.txt
│   ├── diagnostics_full_seed42_bs8.csv
│   ├── diagnostics_full_seed42_bs8_summary.json
│   ├── compile_graph_capture/
│   ├── compile_profiler_full/
│   └── final_full_sst2_seed42/
├── figures/
│   ├── latency_comparison.png
│   ├── throughput_comparison.png
│   ├── model_size_comparison.png
│   ├── accuracy_comparison.png
│   ├── presentation_batch8_latency.png
│   ├── presentation_batch8_throughput.png
│   ├── presentation_model_size.png
│   └── presentation_correct_predictions.png
└── transfer_full_sst2_seed42/
    ├── full_sst2_seed42_part_aa
    ├── full_sst2_seed42_part_ab
    ├── full_sst2_seed42_part_ac
    ├── full_sst2_seed42_part_ad
    ├── full_sst2_seed42_part_ae
    └── full_sst2_seed42_part_af
7. Reconstructing the Trained Checkpoint

The trained checkpoint is too large to store as one normal GitHub file. Therefore, it is included as split archive chunks under:

transfer_full_sst2_seed42/

To reconstruct the checkpoint, run from inside FinalProject:

cd transfer_full_sst2_seed42

cat full_sst2_seed42_part_* > full_sst2_seed42_checkpoint_and_results.tar.gz

tar -xzf full_sst2_seed42_checkpoint_and_results.tar.gz

cd ..

cp -r transfer_full_sst2_seed42/checkpoints .

After reconstruction, the checkpoint should exist at:

checkpoints/our_finetuned_distilbert_sst2_full_seed42

Expected files:

config.json
model.safetensors
tokenizer.json
tokenizer_config.json
training_summary.json
8. Setup Instructions

Create and activate a Python environment:

python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
9. Run Final Benchmark

After reconstructing the checkpoint, run the final batch-8 comparison:

python3 src/benchmark_all.py \
  --model-dir checkpoints/our_finetuned_distilbert_sst2_full_seed42 \
  --batch-sizes 8 \
  --warmup 5 \
  --steps 20 \
  --validation-limit 872 | tee results/local_full_seed42_batch8_compile_log.txt

This benchmarks:

FP32 baseline
Dynamic INT8 quantized model
Compiled FP32 model using torch.compile
10. Run Quantization Diagnostic
python3 src/compare_fp32_int8_predictions.py \
  --model-dir checkpoints/our_finetuned_distilbert_sst2_full_seed42 \
  --batch-size 8 \
  --validation-limit 872 \
  --output-csv results/diagnostics_full_seed42_bs8.csv

This compares FP32 and INT8 predictions example by example.

11. Capture torch.compile Graph
python3 src/capture_compile_graph.py \
  --model-dir checkpoints/our_finetuned_distilbert_sst2_full_seed42 \
  --output-dir results/compile_graph_capture

Important output files:

results/compile_graph_capture/dynamo_explain.txt
results/compile_graph_capture/capture_summary.json
results/compile_graph_capture/graph_0.txt
results/compile_graph_capture/graph_0_code.py
12. Run PyTorch Profiler
python3 src/profile_eager_vs_compile_full.py

Important output files:

results/compile_profiler_full/eager_fp32_profile_table.txt
results/compile_profiler_full/compiled_fp32_profile_table.txt
results/compile_profiler_full/eager_fp32_top_ops.csv
results/compile_profiler_full/compiled_fp32_top_ops.csv
results/compile_profiler_full/compile_profile_summary.json

Profiler summary:

Metric / Operation	Eager FP32	Compiled FP32	Change
Total self CPU time	4.606 s	3.796 s	17.6% lower
aten::copy_	325.20 ms	168.07 ms	48.3% lower
scaled-dot-product attention	227.23 ms	140.39 ms	38.2% lower
Batch-8 latency	435.28 ms	394.05 ms	9.5% lower
13. Important Notes
The early 4,000-example and 8,000-example runs were development/debugging runs.
The final reported model is the full-SST-2 model trained on 67,349 examples.
Final reported metrics are based on the full trained checkpoint.
A100 GPU was used for training.
Local CPU / WSL was used for final batch-8 inference comparison.
Dynamic quantization was evaluated on CPU because PyTorch dynamic quantization is mainly CPU-oriented.
torch.compile changes execution, not model weights.
INT8 quantization improved latency and model size, but caused a small accuracy drop.
14. References
Hugging Face Transformers: https://huggingface.co/docs/transformers
Hugging Face Datasets: https://huggingface.co/docs/datasets
GLUE Benchmark / SST-2: https://gluebenchmark.com/
DistilBERT model: https://huggingface.co/distilbert-base-uncased
PyTorch Quantization: https://pytorch.org/docs/stable/quantization.html
PyTorch torch.compile: https://pytorch.org/docs/stable/generated/torch.compile.html
PyTorch Profiler: https://pytorch.org/docs/stable/profiler.html
BERT paper: https://arxiv.org/abs/1810.04805
DistilBERT paper: https://arxiv.org/abs/1910.01108
