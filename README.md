# Fine-Tune FLAN-T5 with Reinforcement Learning (PPO) to Generate Less-Toxic Summaries

This notebook demonstrates how to fine-tune a FLAN-T5 model using Proximal Policy Optimization (PPO) and Parameter Efficient Fine-Tuning (PEFT) to reduce toxicity in generated summaries.

## Overview

The project uses Meta AI's hate speech reward model as a binary classifier to train FLAN-T5 to generate less toxic content while maintaining summary quality on the DialogSum dataset.

## Requirements

### System Requirements
- **Instance Type**: ml.m5.2xlarge (8 vCPUs, 32 GiB RAM)
- **Python**: 3.12+

### Dependencies
```bash
pip install tensorflow==2.18.0 keras==3.9.0
pip install torch==2.5.1 torchdata==0.6.0
pip install datasets==2.17.0 transformers==4.38.2 evaluate==0.4.0
pip install rouge_score==0.1.2 peft==0.3.0
pip install git+https://github.com/lvwerra/trl.git@25fa1bd
```

## Dataset

- **Source**: [DialogSum](https://huggingface.co/datasets/knkarthick/dialogsum)
- **Preprocessing**: Filters dialogues between 200-1000 characters
- **Split**: 80% train, 20% test

## Model Architecture

- **Base Model**: [google/flan-t5-base](https://huggingface.co/google/flan-t5-base)
- **Fine-tuning**: LoRA (Low-Rank Adaptation)
- **Training Method**: PPO (Proximal Policy Optimization)
- **Reward Model**: Meta AI hate speech classifier

## Key Components

1. **PEFT Model Loading**: Pre-trained summarization adapter
2. **Reward Model**: Binary toxicity classifier
3. **PPO Training**: Reinforcement learning optimization
4. **Evaluation**: Quantitative and qualitative toxicity assessment

## File Structure

```
├── Lab_3_fRLHF_model_to_detoxify_summaries.ipynb
├── peft-dialogue-summary-checkpoint-from-s3/
│   ├── adapter_model.bin
│   ├── adapter_config.json
│   └── tokenizer files
└── README.md
```

## Key Features

- **Toxicity Reduction**: Uses reinforcement learning to minimize hate speech
- **Efficiency**: PEFT reduces trainable parameters to ~1.41% of total model
- **Quality Preservation**: Maintains summarization performance while reducing toxicity
- **Scalable**: Works with standard compute resources

## Expected Outcomes

- Reduced toxicity scores in generated summaries
- Maintained or improved summary quality (ROUGE scores)
- Demonstrable improvement in model safety metrics

## Notes

- Training may take several minutes depending on dataset size
- Ignore warnings during package installation
- Model checkpoints are automatically saved during training"# Detoxify-Dialogue-Summarization-with-RLHF" 
