import logging
from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig

logger = logging.getLogger(__name__)

class AttentionLayer(nn.Module):
    """Multi-head self-attention layer for enhanced text representation."""
    
    def __init__(self, hidden_size: int, num_attention_heads: int = 8, dropout: float = 0.1):
        """
        Initialize attention layer.
        
        Args:
            hidden_size: Size of the input hidden states
            num_attention_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"Hidden size {hidden_size} must be divisible by number of attention heads {num_attention_heads}"
            )

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # Initialize layers
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize the weights using Xavier uniform initialization."""
        for module in [self.query, self.key, self.value, self.dense]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transpose and reshape tensor for attention calculation.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, all_head_size)
            
        Returns:
            Tensor of shape (batch_size, num_attention_heads, seq_length, attention_head_size)
        """
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self, 
        hidden_states: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of attention layer.
        
        Args:
            hidden_states: Input tensor of shape (batch_size, seq_length, hidden_size)
            attention_mask: Optional mask tensor of shape (batch_size, seq_length)
            
        Returns:
            Tensor of shape (batch_size, seq_length, hidden_size)
        """
        # Ensure input has correct shape
        batch_size, seq_length, hidden_size = hidden_states.size()
        
        # Project inputs to query, key, and value
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        # Calculate attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / torch.sqrt(
            torch.tensor(self.attention_head_size, dtype=torch.float, device=hidden_states.device)
        )

        # Apply attention mask if provided
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_scores = attention_scores.masked_fill(attention_mask == 0, float('-inf'))

        # Calculate attention probabilities
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        # Apply attention to values
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # Final dense projection and residual connection
        attention_output = self.dense(context_layer)
        attention_output = self.dropout(attention_output)
        attention_output = self.layer_norm(attention_output + hidden_states)

        return attention_output

class StanceDetectionModel(nn.Module):
    """
    Stance detection model using BERT with enhanced attention and pooling mechanisms.
    """
    
    def __init__(
        self,
        bert_model_name: str = "vinai/bertweet-base",
        num_classes: int = 2,
        dropout: float = 0.4,
        gradient_checkpointing: bool = False
    ):
        """
        Initialize the stance detection model.
        
        Args:
            bert_model_name: Name of the pretrained BERT model to use
            num_classes: Number of output classes
            dropout: Dropout probability
            gradient_checkpointing: Whether to use gradient checkpointing
        """
        super().__init__()

        # Load BERT configuration and model
        self.config = AutoConfig.from_pretrained(bert_model_name)
        self.bert = AutoModel.from_pretrained(bert_model_name, config=self.config)
        
        if gradient_checkpointing:
            self.bert.gradient_checkpointing_enable()

        # Custom attention layer
        self.attention = AttentionLayer(
            hidden_size=self.bert.config.hidden_size,
            num_attention_heads=8,
            dropout=dropout
        )

        # Define classifier dimensions
        classifier_dim = self.bert.config.hidden_size * 3  # CLS + attention + mean
        intermediate_dim = 768
        final_intermediate_dim = 384

        # Enhanced stance classification head with deeper architecture
        self.stance_classifier = nn.Sequential(
            nn.LayerNorm(classifier_dim),
            nn.Dropout(dropout),
            nn.Linear(classifier_dim, intermediate_dim),
            nn.LayerNorm(intermediate_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_dim, final_intermediate_dim),
            nn.LayerNorm(final_intermediate_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(final_intermediate_dim, num_classes)
        )

        # Initialize classifier weights
        self._init_classifier_weights()

    def _init_classifier_weights(self):
        """Initialize the weights of the classifier layers."""
        for name, module in self.stance_classifier.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def _pool_hidden_states(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Pool hidden states using multiple strategies.
        
        Args:
            hidden_states: Output from BERT model
            attention_mask: Attention mask for valid tokens
            
        Returns:
            Tuple of (CLS token output, attention-weighted output, mean-pooled output)
        """
        # Get [CLS] token representation
        cls_output = hidden_states[:, 0, :]
        
        # Attention-weighted pooling
        attention_weights = torch.softmax(
            torch.sum(hidden_states * attention_mask.unsqueeze(-1), dim=-1),
            dim=-1
        ).unsqueeze(-1)
        attention_output = torch.sum(hidden_states * attention_weights, dim=1)
        
        # Mean pooling with mask
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
        sum_mask = mask_expanded.sum(dim=1)
        mean_output = sum_embeddings / (sum_mask + 1e-9)
        
        return cls_output, attention_output, mean_output

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_hidden_states: bool = False
    ) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask for valid tokens
            return_hidden_states: Whether to return hidden states
            
        Returns:
            Model output logits and optionally hidden states
        """
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        hidden_states = outputs.last_hidden_state
        
        # Apply custom attention
        attended_output = self.attention(hidden_states, attention_mask)
        
        # Pool hidden states
        cls_output, attention_output, mean_output = self._pool_hidden_states(
            attended_output,
            attention_mask
        )
        
        # Concatenate different representations
        combined = torch.cat([cls_output, attention_output, mean_output], dim=1)
        
        # Get stance logits
        stance_logits = self.stance_classifier(combined)
        
        if return_hidden_states:
            return stance_logits, hidden_states
        return stance_logits

    def get_attention_weights(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Get attention weights for interpretability.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask for valid tokens
            
        Returns:
            Attention weights
        """
        self.eval()
        with torch.no_grad():
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            hidden_states = outputs.last_hidden_state
            
            # Get attention weights from custom attention layer
            query = self.attention.query(hidden_states)
            key = self.attention.key(hidden_states)
            
            attention_weights = torch.matmul(query, key.transpose(-1, -2))
            attention_weights = attention_weights / torch.sqrt(
                torch.tensor(self.attention.attention_head_size, dtype=torch.float, device=query.device)
            )
            
            if attention_mask is not None:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                attention_weights = attention_weights.masked_fill(
                    attention_mask == 0, float('-inf')
                )
            
            attention_weights = F.softmax(attention_weights, dim=-1)
            
        return attention_weights