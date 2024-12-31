import torch
from stance_model import StanceDetectionModel
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import json
from pathlib import Path
from tqdm import tqdm
import argparse
from transformers import AutoTokenizer
import logging
import seaborn as sns
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('testing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class StanceDataset(Dataset):
    """Dataset for stance detection testing."""
    
    def __init__(
        self,
        texts: List[str],
        topics: List[str],
        labels: List[int],
        tokenizer,
        max_length: int = 128,
        stride: int = 96
    ):
        self.texts = texts
        self.topics = topics
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        topic = str(self.topics[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            topic,
            truncation="only_first",
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
            stride=self.stride,
            return_overflowing_tokens=True
        )

        return {
            "input_ids": encoding["input_ids"][0],
            "attention_mask": encoding["attention_mask"][0],
            "labels": torch.tensor(label, dtype=torch.long)
        }

class StanceTester:
    def __init__(self, model_path: str, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = self.load_model(model_path)
        self.model.to(device)
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", normalization=True)
        self.max_length = 128
        self.stride = 96
        
    def load_model(self, model_path: str) -> StanceDetectionModel:
        """Load the model from checkpoint."""
        try:
            # Initialize model with 2 classes
            model = StanceDetectionModel(
                bert_model_name="vinai/bertweet-base",
                num_classes=2,  # Two classes: Against, Favor
                dropout=0.4
            )
            
            logger.info(f"Loading model from {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])

            logger.info("Model loaded successfully")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def predict_batch(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Make predictions for a batch of data."""
        self.model.eval()
        with torch.no_grad():
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(batch["input_ids"], batch["attention_mask"])
            return torch.argmax(outputs, dim=1)

    def evaluate_test_set(self, test_df: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate model on test dataset"""
        self.model.eval()
        
        # Print detailed label information
        logger.info("\nOriginal label distribution in test data:")
        logger.info(test_df["label"].value_counts().sort_index())
        
        # Ensure only binary labels are considered
        test_df = test_df[test_df["label"].isin([0, 1])]

        logger.info("\nModified label distribution (after ensuring binary labels):")
        logger.info(test_df["label"].value_counts().sort_index())
        
        # Create dataset and dataloader
        dataset = StanceDataset(
            texts=test_df["text"].tolist(),
            topics=test_df["topic"].tolist(),
            labels=test_df["label"].tolist(),
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            stride=self.stride
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=16,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        all_preds = []
        all_labels = []
        
        # Make predictions
        for batch in tqdm(dataloader, desc="Evaluating"):
            preds = self.predict_batch(batch)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["labels"].numpy())
        
        # Use binary classification labels
        target_names = ['Against', 'Favor']

        if len(np.unique(all_labels)) < 2:
            print(f"Warning: Only one unique label found in the data. Returning default values.")
            return {
                'classification_report': {
                    'Against': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 0},
                    'Favor': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 0},
                    'macro avg': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 0},
                    'weighted avg': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 0},
                    'accuracy': 0.0
                },
                'confusion_matrix': [[0, 0], [0, 0]]
            }
        
        # Calculate metrics
        report = classification_report(
            all_labels, 
            all_preds,
            target_names=target_names,
            output_dict=True
        )
        conf_matrix = confusion_matrix(all_labels, all_preds)
        
        return {
            'classification_report': report,
            'confusion_matrix': conf_matrix.tolist() # Convert to list for JSON serialization
        }

    def predict_single_text(self, text: str, topic: str) -> Dict[str, Any]:
        """Make prediction for a single text-topic pair."""
        self.model.eval()
        
        # Create a batch of 1
        encoding = self.tokenizer(
            text,
            topic,
            truncation="only_first",
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
            stride=self.stride,
            return_overflowing_tokens=True
        )
        
        batch = {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"]
        }
        
        # Get prediction
        with torch.no_grad():
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(batch["input_ids"], batch["attention_mask"])
            probs = torch.softmax(outputs, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred].item()
        
        stance_labels = ["Against", "Favor"]
        stance = stance_labels[pred]
        
        return {
            "stance": stance,
            "confidence": confidence,
            "probabilities": {
                label: prob.item()
                for label, prob in zip(stance_labels, probs[0])
            }
        }

def interactive_testing(tester: StanceTester):
    """Interactive testing interface"""
    logger.info("\nWelcome to the Stance Detection Testing Interface!")
    logger.info("Enter 'quit' to exit")
    
    while True:
        text = input("\nEnter text: ")
        if text.lower() == 'quit':
            break
            
        topic = input("Enter topic: ")
        
        # Get prediction
        result = tester.predict_single_text(text, topic)
        
        # Display results
        print(f"\nResults:")
        print(f"Stance: {result['stance']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print("\nProbabilities for each class:")
        for stance, prob in result['probabilities'].items():
            print(f"{stance}: {prob:.4f}")

def plot_confusion_matrix(cm: np.ndarray, labels: List[str], filename: str):
    """Plot and save the confusion matrix as an image."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig(filename)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Test Stance Detection Model')
    parser.add_argument('--model_path', type=str, default='model_outputs/best_model.pt',
                      help='Path to trained model checkpoint')
    parser.add_argument('--test_file', type=str, default='test_stance.csv',
                      help='Path to test dataset')
    parser.add_argument('--mode', type=str, choices=['interactive', 'evaluate', 'both'],
                      default='evaluate', help='Testing mode')
    args = parser.parse_args()
    
    try:
        # Initialize tester
        tester = StanceTester(args.model_path)
        
        # Evaluation mode
        if args.mode in ['evaluate', 'both']:
            logger.info("Evaluating model on test set...")
            test_df = pd.read_csv(args.test_file, sep='|')
            results = tester.evaluate_test_set(test_df)
            
            # Save results
            output_path = Path('evaluation_results.json')
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            # Plot confusion matrix
            cm = np.array(results['confusion_matrix'])
            plot_confusion_matrix(cm, ['Against', 'Favor'], 'confusion_matrix.png')
            
            logger.info("\nTest Set Results:")
            logger.info("\nClassification Report:")
            for label, metrics in results['classification_report'].items():
                if isinstance(metrics, dict):
                    logger.info(f"\n{label}:")
                    logger.info(f"Precision: {metrics['precision']:.4f}")
                    logger.info(f"Recall: {metrics['recall']:.4f}")
                    logger.info(f"F1-score: {metrics['f1-score']:.4f}")
            
            logger.info("\nConfusion Matrix (saved as image): confusion_matrix.png")
        
        # Interactive mode
        if args.mode in ['interactive', 'both']:
            interactive_testing(tester)
            
    except Exception as e:
        logger.error(f"Error during testing: {str(e)}")
        raise

if __name__ == "__main__":
    main()