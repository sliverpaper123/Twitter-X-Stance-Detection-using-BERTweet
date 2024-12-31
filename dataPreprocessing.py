import re
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import contractions
import emoji
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StanceDataPreprocessor:
    def __init__(self, keep_numbers: bool = True, keep_special_chars: bool = True):
        """
        Initialize the preprocessor with configurable options.
        
        Args:
            keep_numbers (bool): Whether to retain numbers in text
            keep_special_chars (bool): Whether to retain special characters
        """
        try:
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('punkt', quiet=True)
            logger.info("NLTK downloads completed.")
        except Exception as e:
            logger.error(f"Error downloading NLTK data: {str(e)}")
            raise

        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))
        self.label_encoder = LabelEncoder()
        self.keep_numbers = keep_numbers
        self.keep_special_chars = keep_special_chars

    def _sanitize_text(self, text: str) -> str:
        """Sanitize text input to prevent injection attacks."""
        # Remove control characters and zero-width spaces
        text = "".join(char for char in text if ord(char) >= 32)
        text = text.replace('\ufeff', '')
        return text

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text with configurable options and better handling.
        
        Args:
            text (str): Input text to preprocess
            
        Returns:
            str: Preprocessed text
        """
        if pd.isna(text):
            return ""

        text = str(text).lower()
        text = self._sanitize_text(text)
        
        # URL removal
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Handle hashtags and mentions
        text = re.sub(r'#(\w+)', r'\1', text)
        text = re.sub(r'@\w+', '@USER', text)  # Replace usernames with token
        
        # Convert emojis to text
        text = emoji.demojize(text)
        
        # Fix contractions
        text = contractions.fix(text)
        
        # Character filtering
        if self.keep_numbers and self.keep_special_chars:
            text = re.sub(r'[^\w\s@#$%-]', '', text)
        elif self.keep_numbers:
            text = re.sub(r'[^\w\s]', '', text)
        elif self.keep_special_chars:
            text = re.sub(r'[^a-zA-Z\s@#$%-]', '', text)
        else:
            text = re.sub(r'[^a-zA-Z\s]', '', text)

        # Tokenization and lemmatization with stopword removal
        tokens = []
        for word in nltk.word_tokenize(text):
            if word and word.lower() not in self.stop_words:
                lemmatized = self.lemmatizer.lemmatize(word)
                if lemmatized:
                    tokens.append(lemmatized)

        return ' '.join(tokens)

    def process_topic(self, topic: str) -> str:
        """
        Process topic with improved handling of special characters and validation.
        
        Args:
            topic (str): Input topic string
            
        Returns:
            str: Processed topic string
        """
        if pd.isna(topic):
            return ""

        topic = str(topic)
        topic = self._sanitize_text(topic)
        
        # Remove quotes but preserve meaningful characters
        topic = topic.replace('"', '').replace("'", '')
        topic = topic.lower().strip()
        
        # Preserve hashtags and special characters that might be meaningful
        topic = re.sub(r'[^\w\s#@-]', '', topic)
        
        return topic

    def validate_data(self, df: pd.DataFrame) -> None:
        """
        Validate input dataframe structure and content.
        
        Args:
            df (pd.DataFrame): Input dataframe to validate
            
        Raises:
            ValueError: If validation fails
        """
        required_columns = ['author', 'post', 'new_topic', 'label']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        if df.empty:
            raise ValueError("DataFrame is empty")
            
        if df['label'].isnull().any():
            raise ValueError("Found null values in label column")

    def preprocess_dataset(
        self, 
        input_file: str, 
        output_file: str, 
        save_stats: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Preprocess entire dataset with improved error handling and statistics.
        
        Args:
            input_file (str): Path to input CSV file
            output_file (str): Path to output CSV file
            save_stats (bool): Whether to save preprocessing statistics
            
        Returns:
            Optional[pd.DataFrame]: Processed dataframe if successful
        """
        try:
            logger.info(f"Reading dataset from {input_file}...")
            input_path = Path(input_file)
            if not input_path.exists():
                raise FileNotFoundError(f"Input file not found: {input_file}")

            df = pd.read_csv(input_file)
            self.validate_data(df)

            logger.info("Processing data...")
            processed_data = {
                "author": df["author"],
                "text": df["post"].apply(self.preprocess_text),
                "topic": df["new_topic"].apply(self.process_topic),
                "label": df["label"]
            }

            df_final = pd.DataFrame(processed_data)
            
            # Save preprocessing statistics
            if save_stats:
                stats = self._generate_stats(df_final)
                stats_file = Path(output_file).with_suffix('.stats.json')
                pd.Series(stats).to_json(stats_file)
                logger.info(f"Statistics saved to {stats_file}")

            logger.info(f"Saving processed dataset to {output_file}...")
            df_final.to_csv(output_file, index=False, sep="|")

            logger.info("\nDataset Statistics:")
            for key, value in stats.items():
                logger.info(f"{key}: {value}")

            return df_final

        except Exception as e:
            logger.error(f"Error during preprocessing: {str(e)}")
            raise

    def _generate_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive statistics about the processed dataset."""
        return {
            "num_samples": len(df),
            "num_unique_topics": df['topic'].nunique(),
            "num_unique_authors": df['author'].nunique(),
            "label_distribution": df['label'].value_counts().to_dict(),
            "avg_text_length": df['text'].str.len().mean(),
            "memory_usage_mb": df.memory_usage().sum() / 1024**2,
        }

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('preprocessing.log'),
            logging.StreamHandler()
        ]
    )

    try:
        # Process training data
        input_file = "stance_dataset_full.csv"
        output_file = "stance_dataset.csv"
        preprocessor = StanceDataPreprocessor(keep_numbers=True, keep_special_chars=True)
        processed_df = preprocessor.preprocess_dataset(input_file, output_file)

        # Process test data
        input_file = "test_stance_full.csv"
        output_file = "test_stance.csv"
        processed_df = preprocessor.preprocess_dataset(input_file, output_file)

    except Exception as e:
        logger.error(f"Preprocessing failed: {str(e)}")
        raise