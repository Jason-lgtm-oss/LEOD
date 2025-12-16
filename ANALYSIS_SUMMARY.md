# LEOD Paper-Code Analysis: Executive Summary

## Project Overview

This directory contains **three comprehensive analysis documents** that provide a complete deep dive into the LEOD paper ("Label-Efficient Object Detection for Event Cameras" - CVPR 2024) and its official PyTorch implementation.

---

## Document Map

### 1. **LEOD_PAPER_CODE_MAPPING.md** (2,100 lines, 71 KB)
**Purpose:** Complete mapping of paper methodology to code implementation

**Contents:**
- Executive summary of LEOD contributions
- Paper core contributions (5 sections)
- Core architecture & system design
- **Method 1: Recurrent Backbone** - Temporal modeling with LSTM + MaxViT
- **Method 2: Pseudo-Label Generation** - TTA, tracking, inpainting
- **Method 3: Data Mixing Strategy** - Random + streaming hybrid approach
- **Method 4: Weakly/Semi-Supervised Learning** - Self-training pipeline
- Evaluation metrics integration (Prophesee AP)
- Experimental configuration & reproduction
- Code quality & innovation highlights
- Training pipeline end-to-end
- Summary table of paper-to-code mappings
- Performance summary from paper

**Best For:**
- Understanding how paper concepts map to code
- Deep understanding of the architecture
- Code review and validation against paper claims

**Key Sections:**
```
Paper: "Recurrent ViT with LSTM"
Code: models/detection/recurrent_backbone/maxvit_rnn.py
      (4 stages with progressive downsampling, LSTM cells)

Paper: "Self-Training with High-Quality Pseudo-Labels"
Code: modules/pseudo_labeler.py
      (TTA → Tracking → Filtering → Inpainting)

Paper: "Mixed Sampling for Training Efficiency"
Code: modules/data/genx.py
      (50% streaming + 50% random access)
```

---

### 2. **IMPLEMENTATION_GUIDE.md** (1,114 lines, 39 KB)
**Purpose:** Deep dive into key modules with code snippets and algorithms

**Contents:**
- **Part 1: Recurrent Backbone Implementation**
  - LSTM state initialization and management
  - Forward pass through recurrent stages
  - Visualizing LSTM state flow
  - ConvLSTM details
  - MaxViT attention mechanism

- **Part 2: Pseudo-Label Generation Pipeline**
  - TTA aggregation with code examples
  - Linear tracking for filtering
  - EventSeqData per-sequence aggregation
  - Complete pseudo-label pipeline

- **Part 3: Data Mixing Strategy**
  - Random-access dataset implementation
  - Streaming dataset implementation
  - Mixed dataset with benefits analysis
  - Batch structure differences

- **Part 4: Self-Training Loop Configuration**
  - Dataset configuration for WSOD/SSOD
  - Model configuration for hard vs soft targets
  - Training hyperparameters by round
  - Step configuration by data ratio

- **Part 5: Quick Reference**
  - Key functions and purposes table
  - Debugging tips (gradient explosion, pseudo-label quality, memory issues)

**Best For:**
- Understanding algorithm details
- Debugging training issues
- Extending the codebase
- Implementing improvements

**Example Content:**
```python
# Detailed explanation of LSTM state detachment
def save_states_and_detach(self, worker_id, states):
    """
    Why detach?
    Gradients flow: loss ← frame100 ← frame99 ← ... ← frame0
    Without detach: gradient explosion, memory explosion
    Solution: Detach at batch boundaries
    """
    detached_states = [h.detach(), c.detach() for h, c in states]
    self.worker_id_2_states[worker_id] = detached_states
```

---

### 3. **REPRODUCTION_GUIDE.md** (832 lines, 21 KB)
**Purpose:** Step-by-step instructions to reproduce all paper results

**Contents:**
- Overview with milestones & timeline
- **Step 1: Environment Setup** (15 min)
  - Conda environment creation
  - Dependency installation
  - Dataset download
  - Pre-trained weights

- **Step 2: Baseline Training** (2 hours)
  - Gen1 & Gen4 WSOD baseline
  - Configuration mapping
  - Code locations
  - Expected results (~28-30% mAP for Gen1)

- **Step 3: Pseudo-Label Generation** (7-10 hours)
  - Algorithm overview
  - Gen1 & Gen4 pseudo-label generation
  - Optional quality evaluation
  - Output structure

- **Step 4: Self-Training Round 1** (2 hours)
  - Dataset config creation
  - Train with soft anchor assignment
  - Expected improvement (~37-38% mAP)

- **Step 5: Final Evaluation** (1 hour)
  - Test set evaluation
  - Visualization generation

- **Step 6: Optional Round 2** (diminishing returns)

- **Complete Checklist** with all steps
- **Troubleshooting** for common issues
- **Expected Results Summary** with tables
- **Key Configuration Decisions**
- **Advanced Experiments & Ablations**

**Best For:**
- Reproducing paper results
- Understanding training pipeline
- Running experiments
- Troubleshooting issues

**Quick Start Example:**
```bash
# Baseline training (Step 2)
python train.py model=rnndet dataset=gen1x0.01_ss \
  +experiment/gen1="small.yaml" training.max_steps=200000

# Pseudo-label generation (Step 3)
python predict.py model=pseudo_labeler dataset=gen1x0.01_ss \
  checkpoint="./ckpts/gen1x0.01_ss/last.ckpt" \
  tta.enable=True save_dir=./datasets/pseudo_gen1/gen1x0.01_ss-1round/train

# Self-training (Step 4)
python train.py model=rnndet-soft dataset=gen1x0.01_ss-1round \
  training.max_steps=150000 training.learning_rate=0.0005
```

---

## Key Concepts Explained Across Documents

### Concept: Recurrent Backbone
- **Paper:** "RNN-based feature extraction with temporal context"
- **PAPER_CODE_MAPPING:** Full architecture description (4 stages, LSTM cells)
- **IMPLEMENTATION_GUIDE:** Detailed code walkthrough with examples
- **REPRODUCTION_GUIDE:** Used in training via `model=rnndet`

### Concept: Pseudo-Label Generation
- **Paper:** "Self-training with TTA, tracking, filtering"
- **PAPER_CODE_MAPPING:** 5-step pipeline (TTA → merge → track → filter → inpaint)
- **IMPLEMENTATION_GUIDE:** Code snippets for each step with algorithms
- **REPRODUCTION_GUIDE:** Step 3 with command line execution

### Concept: Data Mixing Strategy
- **Paper:** "Hybrid random + streaming sampling"
- **PAPER_CODE_MAPPING:** Why mixed sampling works
- **IMPLEMENTATION_GUIDE:** Implementation details with batch structures
- **REPRODUCTION_GUIDE:** Configuration parameters to adjust

### Concept: Self-Training Loop
- **Paper:** "Iterative improvement through pseudo-labels"
- **PAPER_CODE_MAPPING:** How rounds work and diminishing returns
- **IMPLEMENTATION_GUIDE:** Hyperparameters by round
- **REPRODUCTION_GUIDE:** Steps 2-4-6 with actual commands

---

## Quick Navigation Guide

**I want to...**

### Understand the Big Picture
→ Start with **LEOD_PAPER_CODE_MAPPING.md** - Section "Executive Summary"

### Reproduce Paper Results
→ Follow **REPRODUCTION_GUIDE.md** - Steps 1-5 in order

### Debug a Training Issue
→ Go to **IMPLEMENTATION_GUIDE.md** - Part 5 "Debugging Tips"

### Understand RNN State Management
→ Read **IMPLEMENTATION_GUIDE.md** - Part 1.1 "LSTM State Initialization"

### See How Pseudo-Labels Work
→ Check **IMPLEMENTATION_GUIDE.md** - Part 2 "Pseudo-Label Pipeline"

### Understand Data Loading
→ See **IMPLEMENTATION_GUIDE.md** - Part 3 "Data Mixing Strategy"

### Compare with Paper
→ Use **LEOD_PAPER_CODE_MAPPING.md** - Summary table at end

---

## Statistics

| Document | Lines | Size | Topics |
|----------|-------|------|--------|
| LEOD_PAPER_CODE_MAPPING.md | 2,100 | 71 KB | 15 major sections |
| IMPLEMENTATION_GUIDE.md | 1,114 | 39 KB | 5 parts, debugging |
| REPRODUCTION_GUIDE.md | 832 | 21 KB | 6 steps, checklist |
| **Total** | **4,046** | **131 KB** | **Complete analysis** |

---

## Paper Summary

**Title:** LEOD: Label-Efficient Object Detection for Event Cameras

**Problem:** Event cameras produce sparse annotations (4 FPS) but have high temporal resolution (>1000 FPS). Only 0.4% of frames are labeled, leading to sub-optimal performance.

**Solution:** Self-training framework combining:
1. **Recurrent temporal modeling** (LSTM + MaxViT backbone)
2. **High-quality pseudo-label generation** (TTA + tracking + filtering)
3. **Mixed sampling strategy** (random + streaming)
4. **Weakly/Semi-supervised learning** (iterative self-training)

**Results:**
- Gen1 (1% labels): **+30% mAP improvement** (28.5% → 37.6%)
- Gen4 (1% labels): **+67% mAP improvement** (12.3% → 20.5%)
- Achieves **97-99% of full supervision performance**

**Key Insights:**
- Pseudo-label precision strongly correlates with next-round performance
- Tracking effectively filters false positives
- Mixed sampling balances efficiency and temporal understanding
- Self-training plateaus after round 2 (diminishing returns)

---

## Code Structure Overview

```
LEOD Project/
│
├── modules/
│   ├── detection.py          [Lightning module for training/val]
│   ├── pseudo_labeler.py     [Pseudo-label generation]
│   ├── data/genx.py          [Mixed data loading]
│   ├── tracking/             [Linear bbox tracker]
│   └── utils/ssod.py         [Semi-supervised utilities]
│
├── models/
│   └── detection/
│       ├── recurrent_backbone/  [Recurrent MaxViT]
│       ├── yolox/               [Detection head]
│       └── yolox_extension/     [Extended head with soft targets]
│
├── data/
│   └── genx_utils/           [Dataset classes & utilities]
│
├── config/
│   ├── experiment/           [Experiment presets (Gen1/Gen4)]
│   ├── dataset/              [Dataset configs (WSOD/SSOD ratios)]
│   ├── model/                [Model configs]
│   └── general.yaml          [Global settings]
│
├── train.py                  [Main training script]
├── predict.py                [Pseudo-label generation script]
├── val.py                    [Evaluation script]
└── vis_pred.py               [Visualization script]
```

---

## Implementation Timeline

| Phase | Duration | Key Action | Output |
|-------|----------|-----------|--------|
| Setup | 15 min | Install deps, download data | Ready to train |
| Round 0 | 2h | Train on 1% GT labels | ~28-30% mAP |
| Gen Pseudo | 7-10h | Generate pseudo-labels | 100-250 MB labels |
| Round 1 | 2h | Train on mixed labels | ~37-38% mAP |
| Evaluate | 1h | Test on test set | Final metrics |
| **Total** | **12-14h** | **Complete pipeline** | **SOTA results** |

---

## Key Technologies

- **PyTorch 2.0** - Deep learning framework
- **PyTorch Lightning 1.8** - Training framework
- **Hydra 1.3** - Configuration management
- **WandB** - Experiment tracking
- **Detectron2** - COCO evaluation metrics
- **Prophesee Toolbox** - Event camera benchmarks

---

## For Different Audiences

### For Researchers
- **Read:** LEOD_PAPER_CODE_MAPPING.md (understand novelty)
- **Then:** IMPLEMENTATION_GUIDE.md Part 1 (recurrent architecture)
- **Then:** Explore codebase with this knowledge

### For ML Engineers
- **Read:** IMPLEMENTATION_GUIDE.md (code details)
- **Then:** REPRODUCTION_GUIDE.md (run experiments)
- **Then:** Modify configs for your use case

### For Practitioners
- **Read:** REPRODUCTION_GUIDE.md (quick start)
- **Then:** Run commands in order
- **Then:** Refer to IMPLEMENTATION_GUIDE.md for issues

### For Students
- **Read:** LEOD_PAPER_CODE_MAPPING.md (concept mapping)
- **Then:** IMPLEMENTATION_GUIDE.md Part 2 (algorithms)
- **Then:** REPRODUCTION_GUIDE.md (hands-on practice)

---

## Key Takeaways

1. **Recurrent backbones** are effective for temporal event data
2. **Pseudo-label quality** matters more than quantity
3. **Mixed sampling** provides best balance of speed and understanding
4. **Self-training** can close 97% of the gap to full supervision
5. **Proper state management** is critical for RNN training

---

## Next Steps

1. **Read:** LEOD_PAPER_CODE_MAPPING.md for big picture
2. **Run:** REPRODUCTION_GUIDE.md Step 1 for setup
3. **Execute:** Steps 2-4 for full pipeline
4. **Explore:** IMPLEMENTATION_GUIDE.md for modifications
5. **Extend:** Implement your own improvements

---

## Contributing

To extend this analysis:
- Add more detailed code comments to source files
- Implement suggested improvements (curriculum learning, uncertainty estimation)
- Test on custom event camera datasets
- Benchmark performance variations

---

**Total Analysis Created: 4,046 lines, 131 KB of comprehensive documentation**

*Created: December 2024*
*For LEOD (CVPR 2024) PyTorch Implementation*
