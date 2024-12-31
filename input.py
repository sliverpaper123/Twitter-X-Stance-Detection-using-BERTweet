import torch
from stance_model import StanceDetectionModel
import pandas as pd
from transformers import AutoTokenizer
import logging
from dataPreprocessing import StanceDataPreprocessor
from difflib import get_close_matches
from typing import List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('input_prediction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class StancePredictor:
    def __init__(self, model_path: str, known_topics: List[str], device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = self.load_model(model_path)
        self.model.to(device)
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", normalization=True)
        self.max_length = 128
        self.stride = 96
        self.preprocessor = StanceDataPreprocessor(keep_numbers=True, keep_special_chars=True)
        self.known_topics = known_topics

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

    def find_closest_topic(self, topic: str) -> str:
        """Find the closest matching topic from the known topics."""
        closest_match = get_close_matches(topic, self.known_topics, n=1, cutoff=0.6)
        if closest_match:
            return closest_match[0]
        else:
            logger.warning(f"No close match found for topic '{topic}'. Using the original topic.")
            return topic

    def predict_single_text(self, text: str, topic: str) -> dict:
        """Make prediction for a single text-topic pair."""
        self.model.eval()

        # Preprocess text and topic
        processed_text = self.preprocessor.preprocess_text(text)
        
        # Find the closest topic
        closest_topic = self.find_closest_topic(topic)
        processed_topic = self.preprocessor.process_topic(closest_topic)

        # Create a batch of 1
        encoding = self.tokenizer(
            processed_text,
            processed_topic,
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
            },
            "original_topic": topic,
            "matched_topic": closest_topic
        }

def interactive_prediction(predictor: StancePredictor):
    """Interactive prediction interface"""
    logger.info("\nWelcome to the Stance Detection Prediction Interface!")
    logger.info("Enter 'quit' to exit")

    while True:
        text = input("\nEnter text: ")
        if text.lower() == 'quit':
            break

        topic = input("Enter topic: ")

        # Get prediction
        result = predictor.predict_single_text(text, topic)

        # Display results
        print(f"\nResults:")
        print(f"Stance: {result['stance']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Original topic: {result['original_topic']}")
        print(f"Matched topic: {result['matched_topic']}")
        print("\nProbabilities for each class:")
        for stance, prob in result['probabilities'].items():
            print(f"{stance}: {prob:.4f}")

if __name__ == "__main__":
    MODEL_PATH = "model_outputs/best_model.pt"  # Replace with your model path
    TRAIN_DATA_PATH = "stance_dataset.csv"  # Replace with your training data path

    try:
        # Load known topics from the training dataset
        train_df = pd.read_csv(TRAIN_DATA_PATH, sep="|")
        known_topics = train_df["topic"].unique().tolist()

        predictor = StancePredictor(MODEL_PATH, known_topics)
        interactive_prediction(predictor)
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise