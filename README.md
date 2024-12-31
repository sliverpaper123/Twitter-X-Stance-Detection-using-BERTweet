# Twitter/X Stance Detection using BERTweet and multi-strategy pooling.

A deep learning-based approach for detecting stance (pro/con) in Twitter data using BERTweet and multi-strategy pooling.

## Overview

This project implements a stance detection system that automatically determines whether a tweet's author is in favor of or against a specific topic. The system uses a fine-tuned BERTweet model with custom attention mechanisms and multi-strategy pooling to achieve high accuracy in stance classification.

## Key Features

- **Efficient Preprocessing Pipeline**: Handles Twitter-specific elements (hashtags, mentions, emojis) while preserving semantic information
- **Advanced Model Architecture**: BERTweet-based model with multi-head attention and multi-strategy pooling
- **Binary Stance Classification**: Classifies tweets as either "pro" or "con" towards given topics
- **High Performance**: Achieves 73.83% accuracy and 74.51% F1-score on test dataset

## Model Architecture

The system consists of three main components:

1. **Preprocessing Module**

   - Text normalization
   - Emoji interpretation
   - Hashtag segmentation
   - URL and mention handling

2. **Stance Detection Model**

   - BERTweet base model
   - Custom attention layer
   - Multi-strategy pooling
   - Stance classification head

3. **Training Pipeline**
   - Mixed precision training
   - Gradient accumulation
   - Learning rate warmup
   - Early stopping

## Requirements

```
torch>=1.9.0
transformers
nltk
emoji
contractions
pandas
numpy
tqdm
optuna
```

## Installation

```bash
git clone https://github.com/username/twitter-stance-detection
cd twitter-stance-detection
pip install -r requirements.txt
```

## Usage

### Data Preprocessing

```python
from preprocessing import StanceDataPreprocessor

preprocessor = StanceDataPreprocessor(
    keep_numbers=True,
    keep_special_chars=True
)

# Process single text
processed_text = preprocessor.preprocess_text(text)

# Process dataset
preprocessor.preprocess_dataset(
    input_file="data/raw/tweets.csv",
    output_file="data/processed/processed_tweets.csv"
)
```

### Training

```python
from train_stance import train_model
from config import TrainingConfig

config = TrainingConfig(
    model_name="vinai/bertweet-base",
    num_classes=2,
    batch_size=8,
    learning_rate=2e-5,
    num_epochs=5
)

model, metrics = train_model(config)
```

### Inference

```python
from test_stance import StanceTester

tester = StanceTester(model_path="model_outputs/best_model.pt")
result = tester.predict_single_text(
    text="This is a great initiative!",
    topic="climate action"
)
```

## Performance

| Metric    | Value  |
| --------- | ------ |
| Accuracy  | 73.83% |
| Precision | 75%    |
| Recall    | 73%    |
| F1-Score  | 74.51% |

*train accuracy

## Citations

If you use this code, please cite our work:

```bibtex
@misc{stance2024twitter,
  title={Twitter Stance Detection Using Fine-tuned BERTweet},
  author={Singareddy, Anish Paul and Devireddy, Rishikesh and Sai, Akula Akshay},
  year={2024},
  publisher={GitHub},
  howpublished={\url{https://github.com/sliverpaper123/twitter-stance-detection}}
}
```



## Acknowledgments

- [BERTweet](https://github.com/VinAIResearch/BERTweet) team for the pre-trained model
- Hugging Face team for the Transformers library
- All contributors to the open-source libraries used in this project

## Contributors

- Anish Paul Singareddy (SE21UCSE018)
- Rishikesh Devireddy (SE21UARI116)
- Akula Akshay Sai (SE21UCSE014)
