import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List
import pandas as pd
from tqdm import tqdm
from test_stance import StanceDataset, StanceTester
from torch.utils.data import DataLoader
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('topic_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def evaluate_all_topics(
    tester: StanceTester,
    test_df: pd.DataFrame
) -> dict[str, float]:
    """
    Evaluates the model's performance on all topics and returns the F1-scores.

    Args:
        tester: The StanceTester object.
        test_df: The test dataframe.

    Returns:
        A dictionary containing F1-scores for each topic.
    """
    # Ensure only binary labels are considered
    test_df = test_df[test_df["label"].isin([0, 1])]

    all_topics = test_df["new_topic"].unique()
    topic_f1_scores = {}

    for topic in tqdm(all_topics, desc="Evaluating all topics"):
        topic_df = test_df[test_df["new_topic"] == topic]

        if len(topic_df) == 0:
            logger.warning(f"No data found for topic: {topic}")
            continue

        dataset = StanceDataset(
            texts=topic_df["post"].tolist(),
            topics=topic_df["new_topic"].tolist(),
            labels=topic_df["label"].tolist(),
            tokenizer=tester.tokenizer,
            max_length=tester.max_length,
            stride=tester.stride
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

        for batch in dataloader:
            preds = tester.predict_batch(batch)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())

        if len(np.unique(all_labels)) < 2:
            logger.warning(f"Only one unique label found for topic '{topic}'. Skipping F1 calculation.")
            f1 = 0.0  # or np.nan, or any other default value
        else:
            from sklearn.metrics import f1_score
            f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)

        topic_f1_scores[topic] = f1

    return topic_f1_scores

def plot_top_f1_scores(topic_f1_scores: dict[str, float], num_top_topics: int, filename: str):
    """
    Plots and saves a bar chart of the top N F1-scores.

    Args:
        topic_f1_scores: A dictionary containing F1-scores for each topic.
        num_top_topics: The number of top topics to plot.
        filename: The filename to save the plot.
    """
    # Sort the topics by F1-score in descending order
    sorted_topics = sorted(topic_f1_scores.items(), key=lambda item: item[1], reverse=True)

    # Get the top N topics and their F1-scores
    top_topics = sorted_topics[:num_top_topics]
    topics, f1_scores = zip(*top_topics)
    
    f1_scores = [score * 100 for score in f1_scores]

    plt.figure(figsize=(12, 6))
    plt.bar(topics, f1_scores, color='skyblue')
    plt.xlabel("Topics", fontsize=14)
    plt.ylabel("F1-Score (%)", fontsize=14)
    plt.title(f"Top {num_top_topics} F1-Scores Across Different Topics", fontsize=16)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(0, 100)

    # Annotate bars with their values
    for i, v in enumerate(f1_scores):
        plt.text(i, v + 2, f"{v:.0f}", ha='center', va='bottom', fontsize=12)

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    logger.info(f"Top {num_top_topics} F1-scores plot saved to {filename}")

def main():
    MODEL_PATH = "model_outputs/best_model.pt"
    TEST_DATA_PATH = "test_stance_full.csv"
    NUM_TOP_TOPICS = 5

    try:
        tester = StanceTester(MODEL_PATH)
        test_df = pd.read_csv(TEST_DATA_PATH)

        all_topic_f1_scores = evaluate_all_topics(tester, test_df)

        logger.info("\nAll Topic F1-scores:")
        for topic, f1 in all_topic_f1_scores.items():
            logger.info(f"{topic}: {f1:.4f}")

        plot_top_f1_scores(all_topic_f1_scores, NUM_TOP_TOPICS, "top_f1_scores.png")

    except Exception as e:
        logger.error(f"Error during topic analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main()