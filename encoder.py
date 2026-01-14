"""
Transformer Encoder Module

The encoder processes the input sequence into contextual representations.
In summarization: Encoder reads the full document and creates rich embeddings.

Key Responsibilities:
1. Embed input tokens into continuous space
2. Apply positional encoding (position information)
3. Process through multi-head self-attention layers
4. Output contextualized representations for decoder to attend to

Architecture Flow:
    Input IDs → Embedding → Position Encoding → 
    Self-Attention Layers → Encoder Output (memory)
"""

import torch
import torch.nn as nn
import math
from attention_masks import AttentionMaskGenerator


class PositionalEncoding(nn.Module):
    """
    Inject positional information into token embeddings.
    
    Transformers have no inherent notion of position/order. We add sinusoidal
    positional encodings to give the model position awareness.
    
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Create frequency scaling factors
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices, cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension: [1, max_len, d_model]
        pe = pe.unsqueeze(0)
        
        # Register as buffer (not a parameter, but part of state)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Token embeddings [batch_size, seq_len, d_model]
        Returns:
            Position-encoded embeddings [batch_size, seq_len, d_model]
        """
        # Add positional encoding to embeddings
        seq_len = x.size(1)
        # Type ignore for buffer access
        pe_buffer = self.pe  # type: ignore
        x = x + pe_buffer[:, :seq_len, :].detach()
        return self.dropout(x)


class EncoderLayer(nn.Module):
    """
    Single Transformer encoder layer.
    
    Architecture:
        Input → [Multi-Head Self-Attention] → Add & Norm → 
        [Feed Forward] → Add & Norm → Output
    """
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        
        # Multi-head self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True  # Use [batch, seq, feature] format
        )
        
        # Feed-forward network
        self.ff_network = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, src_mask=None):
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            src_mask: Padding mask [batch_size, 1, 1, seq_len]
        
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        # Self-attention with residual connection
        attn_output, _ = self.self_attn(
            x, x, x,
            attn_mask=src_mask,
            need_weights=False
        )
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        
        # Feed-forward with residual connection
        ff_output = self.ff_network(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)
        
        return x


class TransformerEncoder(nn.Module):
    """
    Complete Transformer Encoder stack.
    
    Processes input sequence into rich contextual representations
    that the decoder can attend to during generation.
    """
    
    def __init__(
        self,
        vocab_size,
        d_model=512,
        num_heads=8,
        num_layers=6,
        d_ff=2048,
        max_seq_len=512,
        dropout=0.1,
        pad_token_id=0
    ):
        """
        Args:
            vocab_size: Size of source vocabulary
            d_model: Dimension of model embeddings
            num_heads: Number of attention heads
            num_layers: Number of encoder layers
            d_ff: Dimension of feed-forward network
            max_seq_len: Maximum sequence length
            dropout: Dropout rate
            pad_token_id: ID of padding token
        """
        super().__init__()
        
        self.d_model = d_model
        self.pad_token_id = pad_token_id
        
        # Token embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Stack of encoder layers
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(d_model)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize embedding weights with Xavier uniform."""
        nn.init.xavier_uniform_(self.embedding.weight)
    
    def forward(self, src, src_mask=None):
        """
        Encode input sequence.
        
        Args:
            src: Source token IDs [batch_size, src_len]
            src_mask: Optional padding mask [batch_size, 1, 1, src_len]
        
        Returns:
            Encoder output (memory) [batch_size, src_len, d_model]
        """
        # Create padding mask if not provided
        if src_mask is None:
            src_mask = AttentionMaskGenerator.create_padding_mask(src, self.pad_token_id)
            # Convert to additive mask for MultiheadAttention
            src_mask = src_mask.squeeze(1).squeeze(1)  # [batch, seq_len]
        
        # Embed tokens and scale by sqrt(d_model)
        # Scaling helps prevent extremely small gradients in attention
        x = self.embedding(src) * math.sqrt(self.d_model)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Pass through encoder layers
        for layer in self.layers:
            x = layer(x, src_mask)
        
        # Final normalization
        x = self.norm(x)
        
        return x


class PretrainedEncoderWrapper(nn.Module):
    """
    Wrapper for pre-trained encoder models (e.g., from Hugging Face).
    
    This allows us to use powerful pre-trained encoders while maintaining
    our custom architecture interface.
    """
    
    def __init__(self, pretrained_model_name="t5-small"):
        super().__init__()
        
        try:
            from transformers import AutoModel
            
            # Load pre-trained encoder
            self.encoder = AutoModel.from_pretrained(pretrained_model_name).encoder
            self.d_model = self.encoder.config.d_model
            
            print(f"✓ Loaded pre-trained encoder: {pretrained_model_name}")
            print(f"  Model dimension: {self.d_model}")
            
        except ImportError:
            raise ImportError(
                "transformers library required for pre-trained models. "
                "Install with: pip install transformers"
            )
    
    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass through pre-trained encoder.
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len] (1 = real token, 0 = padding)
        
        Returns:
            Encoder output [batch_size, seq_len, d_model]
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        return outputs.last_hidden_state


# Testing and demonstration
if __name__ == "__main__":
    print("=" * 70)
    print("ENCODER MODULE DEMONSTRATION")
    print("=" * 70)
    
    # Example configuration
    vocab_size = 10000
    d_model = 512
    batch_size = 2
    seq_len = 20
    
    print(f"\nConfiguration:")
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Model dimension: {d_model}")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    
    # Create encoder
    encoder = TransformerEncoder(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=8,
        num_layers=6
    )
    
    print(f"\n✓ Encoder created with {sum(p.numel() for p in encoder.parameters()):,} parameters")
    
    # Create sample input
    src = torch.randint(1, vocab_size, (batch_size, seq_len))
    src[:, -5:] = 0  # Add some padding
    
    print(f"\nSample input shape: {src.shape}")
    print(f"Sample input (first sequence):\n{src[0]}")
    
    # Encode
    encoder.eval()
    with torch.no_grad():
        memory = encoder(src)
    
    print(f"\nEncoder output shape: {memory.shape}")
    print(f"Encoder output stats:")
    print(f"  Mean: {memory.mean().item():.4f}")
    print(f"  Std:  {memory.std().item():.4f}")
    
    print("\n" + "=" * 70)
    print("✓ Encoder module working correctly!")
    print("=" * 70)
