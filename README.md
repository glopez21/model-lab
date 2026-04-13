# Model Lab

Experimental workspace for building and fine-tuning summarization models.

## Structure

- `models/` - Custom model implementations (drop-in agents for phase3)
- `data/` - Training and evaluation datasets
- `notebooks/` - Jupyter notebooks for experimentation
- `scripts/` - Training, fine-tuning, and evaluation scripts
- `configs/` - Model and training configuration files
- `results/` - Training logs, metrics, and saved model artifacts

## Models

| Model | Base | Description | Status |
|-------|------|-------------|--------|
| `pegasus_summarizer` | google/pegasus-cnn_dailymail | Pegasus-based summarizer | Experimental |
| `led_summarizer` | allenai/led-base-16384 | Long-document summarizer | Experimental |
| `cyber_summarizer` | fine-tuned BART | Cybersecurity-domain fine-tune | Planned |

## Quick Start

1. Copy a model template from `models/templates/`
2. Customize the model name and config
3. Fine-tune using scripts in `scripts/`
4. Drop the agent into `production-plan/phase3/agents/` and register in `dashboard.py`

## Development

```bash
cd model-lab
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```