import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import json
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup
)

from stance_model import StanceDetectionModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Configuration for training."""
    model_name: str = "vinai/bertweet-base"
    num_classes: int = 2
    max_length: int = 128
    stride: int = 96
    train_batch_size: int = 8
    eval_batch_size: int = 16
    num_epochs: int = 20
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    dropout: float = 0.4
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class StanceDataset(Dataset):
    """Dataset for stance detection."""
    
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

        # Changed tokenizer parameters to fix the truncation issue
        encoding = self.tokenizer(
            text,
            topic,
            truncation="only_first",  # Changed from longest_first
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

class EarlyStopping:
    """Early stopping handler."""
    
    def __init__(self, patience: int = 3, min_delta: float = 0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.early_stop

class MetricTracker:
    """Track and compute various metrics."""
    
    def __init__(self, num_classes: int = 2):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.predictions = []
        self.targets = []
        self.running_loss = 0
        self.count = 0

    def update(self, preds: torch.Tensor, targets: torch.Tensor, loss: float):
        self.predictions.extend(preds.cpu().numpy())
        self.targets.extend(targets.cpu().numpy())
        self.running_loss += loss
        self.count += 1

    def compute(self) -> Dict[str, float]:
        from sklearn.metrics import precision_recall_fscore_support, accuracy_score
        
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets,
            predictions,
            average='binary',
            zero_division=0
        )
        
        return {
            'loss': self.running_loss / self.count,
            'accuracy': accuracy_score(targets, predictions),
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_and_preprocess_data(
    data_path: str,
    val_size: float = 0.1
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and preprocess the stance detection dataset."""
    logger.info(f"Loading data from {data_path}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
        
    try:
        df = pd.read_csv(data_path, sep="|")
    except Exception as e:
        logger.error(f"Error reading data file: {e}")
        raise

    # Validate data
    required_columns = ['text', 'topic', 'label']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Missing required columns: {required_columns}")

    # Ensure binary labels
    df = df[df["label"].isin([0, 1])]

    # Split data
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(
        df,
        test_size=val_size,
        stratify=df['label'],
        random_state=42
    )

    logger.info(f"Training samples: {len(train_df)}, Validation samples: {len(val_df)}")
    logger.info(f"Training class distribution:\n{train_df['label'].value_counts(normalize=True)}")
    logger.info(f"Validation class distribution:\n{val_df['label'].value_counts(normalize=True)}")

    return train_df, val_df

def train_epoch(
    model: StanceDetectionModel,
    dataloader: DataLoader,
    optimizer: AdamW,
    scheduler,
    config: TrainingConfig,
    scaler: GradScaler,
    class_weights: Optional[torch.Tensor] = None
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    metric_tracker = MetricTracker()

    progress_bar = tqdm(dataloader, desc="Training")
    for step, batch in enumerate(progress_bar):
        # Move batch to device
        batch = {k: v.to(config.device) for k, v in batch.items()}

        # Forward pass with mixed precision
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):  # Fixed autocast
            outputs = model(batch["input_ids"], batch["attention_mask"])
            if class_weights is not None:
                loss = F.cross_entropy(outputs, batch["labels"], weight=class_weights)
            else:
                loss = F.cross_entropy(outputs, batch["labels"])
            loss = loss / config.gradient_accumulation_steps

        # Backward pass
        scaler.scale(loss).backward()

        if (step + 1) % config.gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config.max_grad_norm
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        # Update metrics
        with torch.no_grad():
            preds = torch.argmax(outputs, dim=1)
            metric_tracker.update(preds, batch["labels"], loss.item())

        # Update progress bar
        progress_bar.set_postfix(
            loss=metric_tracker.running_loss / (step + 1),
            lr=scheduler.get_last_lr()[0]
        )

    return metric_tracker.compute()

@torch.no_grad()
def evaluate(
    model: StanceDetectionModel,
    dataloader: DataLoader,
    config: TrainingConfig,
    class_weights: Optional[torch.Tensor] = None
) -> Dict[str, float]:
    """Evaluate the model."""
    model.eval()
    metric_tracker = MetricTracker()

    for batch in tqdm(dataloader, desc="Evaluating"):
        batch = {k: v.to(config.device) for k, v in batch.items()}
        outputs = model(batch["input_ids"], batch["attention_mask"])
        
        if class_weights is not None:
            loss = F.cross_entropy(outputs, batch["labels"], weight=class_weights)
        else:
            loss = F.cross_entropy(outputs, batch["labels"])

        preds = torch.argmax(outputs, dim=1)
        metric_tracker.update(preds, batch["labels"], loss.item())

    return metric_tracker.compute()

def save_checkpoint(
    model: StanceDetectionModel,
    optimizer: AdamW,
    scheduler,
    metrics: Dict[str, float],
    epoch: int,
    path: str
):
    """Save model checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'metrics': metrics,
    }, path)

def load_checkpoint(
    model: StanceDetectionModel,
    optimizer: Optional[AdamW],
    scheduler: Optional[object],
    path: str
) -> Dict:
    """Load model checkpoint."""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint

def plot_training_history(metrics_history: List[Dict], save_path: str):
    """Plot training metrics history."""
    metrics_df = pd.DataFrame(metrics_history)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training History')
    
    # Loss plot
    axes[0, 0].plot(metrics_df['train_loss'], label='Train')
    axes[0, 0].plot(metrics_df['val_loss'], label='Validation')
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].legend()
    
    # F1 Score plot
    axes[0, 1].plot(metrics_df['train_f1'], label='Train')
    axes[0, 1].plot(metrics_df['val_f1'], label='Validation')
    axes[0, 1].set_title('F1 Score')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].legend()
    
    # Precision plot
    axes[1, 0].plot(metrics_df['train_precision'], label='Train')
    axes[1, 0].plot(metrics_df['val_precision'], label='Validation')
    axes[1, 0].set_title('Precision')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].legend()
    
    # Recall plot
    axes[1, 1].plot(metrics_df['train_recall'], label='Train')
    axes[1, 1].plot(metrics_df['val_recall'], label='Validation')
    axes[1, 1].set_title('Recall')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def train_model(config: TrainingConfig):
    """Main training function."""
    logger.info("Starting training with configuration:")
    logger.info(config)
    
    # Set seeds for reproducibility
    set_seed(config.seed)
    
    # Create output directory
    output_dir = Path("model_outputs")
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    train_df, val_df = load_and_preprocess_data("stance_dataset.csv")
    
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, normalization=True)
    model = StanceDetectionModel(
        bert_model_name=config.model_name,
        num_classes=config.num_classes,
        dropout=config.dropout
    ).to(config.device)
    
    # Create datasets and dataloaders
    train_dataset = StanceDataset(
        texts=train_df["text"].tolist(),
        topics=train_df["topic"].tolist(),
        labels=train_df["label"].tolist(),
        tokenizer=tokenizer,
        max_length=config.max_length,
        stride=config.stride
    )
    
    val_dataset = StanceDataset(
        texts=val_df["text"].tolist(),
        topics=val_df["topic"].tolist(),
        labels=val_df["label"].tolist(),
        tokenizer=tokenizer,
        max_length=config.max_length,
        stride=config.stride
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Calculate class weights
    class_counts = train_df['label'].value_counts()
    class_weights = torch.tensor(
        [1.0 / count for count in class_counts],
        dtype=torch.float32,
        device=config.device
    )
    class_weights = class_weights / class_weights.sum()
    
    # Initialize optimizer, scheduler, and scaler
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    num_training_steps = len(train_dataloader) * config.num_epochs
    num_warmup_steps = int(num_training_steps * config.warmup_ratio)
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # Fixed GradScaler initialization
    scaler = torch.cuda.amp.GradScaler()
    early_stopping = EarlyStopping(patience=3)
    
    # Training loop
    metrics_history = []
    best_val_f1 = 0.0
    
    for epoch in range(config.num_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{config.num_epochs}")
        
        # Training phase
        train_metrics = train_epoch(
            model,
            train_dataloader,
            optimizer,
            scheduler,
            config,
            scaler,
            class_weights
        )
        
        # Validation phase
        val_metrics = evaluate(
            model,
            val_dataloader,
            config,
            class_weights
        )
        
        # Log metrics
        epoch_metrics = {
            'epoch': epoch + 1,
            'train_loss': train_metrics['loss'],
            'val_loss': val_metrics['loss'],
            'train_f1': train_metrics['f1'],
            'val_f1': val_metrics['f1'],
            'train_precision': train_metrics['precision'],
            'val_precision': val_metrics['precision'],
            'train_recall': train_metrics['recall'],
            'val_recall': val_metrics['recall']
        }
        metrics_history.append(epoch_metrics)
        
        # Save metrics and plot
        with open(output_dir / "metrics_history.json", 'w') as f:
            json.dump(metrics_history, f, indent=2)
            
        plot_training_history(metrics_history, output_dir / "training_history.png")
        
        # Log current metrics
        logger.info(f"\nValidation Metrics:")
        for metric_name, value in val_metrics.items():
            logger.info(f"{metric_name}: {value:.4f}")
        
        # Save best model
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                val_metrics,
                epoch,
                output_dir / "best_model.pt"
            )
            logger.info(f"New best model saved! F1: {best_val_f1:.4f}")
        
        # Early stopping check
        if early_stopping(val_metrics['loss']):
            logger.info("Early stopping triggered!")
            break
        
        # Clear GPU memory
        torch.cuda.empty_cache()
    
    logger.info("Training completed!")
    return model, metrics_history

def objective(trial: optuna.Trial) -> float:
    """Objective function for hyperparameter optimization."""
    config = TrainingConfig(
        learning_rate=trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),  # Narrower range
        weight_decay=trial.suggest_float("weight_decay", 0.01, 0.1),              # Narrower range
        dropout=trial.suggest_float("dropout", 0.1, 0.5),
        warmup_ratio=trial.suggest_float("warmup_ratio", 0.1, 0.2),
        max_length=trial.suggest_categorical("max_length", [128]),                # Fixed for stability
        stride=trial.suggest_categorical("stride", [96]),                         # Fixed for stability
        gradient_accumulation_steps=trial.suggest_categorical(
            "gradient_accumulation_steps",
            [4]                                                                   # Fixed for stability
        ),
        train_batch_size=8,                                                      # Fixed batch size
        num_epochs=20                                                             # Fixed epochs
    )
    
    try:
        model = StanceDetectionModel(
            bert_model_name=config.model_name,
            num_classes=config.num_classes,
            dropout=config.dropout
        )
        
        # Initialize model weights before training
        model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
        
        _, metrics_history = train_model(config)
        best_val_f1 = max(m['val_f1'] for m in metrics_history)
        return best_val_f1
    except Exception as e:
        logger.error(f"Trial failed: {str(e)}")
        raise optuna.exceptions.TrialPruned()

if __name__ == "__main__":
    logger.info("Starting stance detection training pipeline...")
    
    # Create study and optimize
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=1, timeout=72000)  # 20 hours timeout
    
    logger.info("Best trial:")
    trial = study.best_trial
    logger.info(f"  Value: {trial.value}")
    logger.info("  Params:")
    for key, value in trial.params.items():
        logger.info(f"    {key}: {value}")
    
    # Train final model with best parameters
    best_config = TrainingConfig(**trial.params)
    final_model, final_metrics = train_model(best_config)
    
    logger.info("Training pipeline completed!")