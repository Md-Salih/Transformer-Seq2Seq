"""
Transformer Decoder Module

The decoder performs AUTOREGRESSIVE generation, producing output one token at a time.
In summarization: Decoder generates summary tokens sequentially, attending to encoder memory.

Key Responsibilities:
1. Generate output tokens autoregressively (one by one)
2. Apply CAUSAL MASKING to prevent looking at future tokens
3. Cross-attend to encoder output (memory) for source context
4. Self-attend to previously generated tokens

CRITICAL CONCEPT: Autoregressive Generation
-------------------------------------------
During inference, the decoder generates token-by-token:
    t0 = <start>
    t1 = decode([t0])
    t2 = decode([t0, t1])
    t3 = decode([t0, t1, t2])
    ...
    
Each new token only sees previous tokens, never future ones.
This is enforced by CAUSAL MASKING during training.
"""

import torch
import torch.nn as nn
import math
from attention_masks import AttentionMaskGenerator, AttentionMaskConverter


class DecoderLayer(nn.Module):
    """
    Single Transformer decoder layer.
    
    Architecture (3 sub-layers):
        1. Masked Self-Attention (causal) - attend to previous decoder tokens
        2. Cross-Attention - attend to encoder output
        3. Feed-Forward Network
        
    Each sub-layer has residual connection + layer normalization.
    """
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        
        # Self-attention (masked/causal)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Cross-attention (to encoder output)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
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
        self.norm3 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, memory, tgt_mask=None, memory_mask=None):
        """
        Args:
            x: Decoder input [batch_size, tgt_len, d_model]
            memory: Encoder output [batch_size, src_len, d_model]
            tgt_mask: Causal mask for self-attention [batch_size, tgt_len, tgt_len]
            memory_mask: Padding mask for encoder output
        
        Returns:
            Output tensor [batch_size, tgt_len, d_model]
        """
        # 1. Masked self-attention (causal)
        attn_output, _ = self.self_attn(
            x, x, x,
            attn_mask=tgt_mask,
            need_weights=False
        )
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        
        # 2. Cross-attention to encoder
        attn_output, _ = self.cross_attn(
            x, memory, memory,
            key_padding_mask=memory_mask,
            need_weights=False
        )
        x = x + self.dropout(attn_output)
        x = self.norm2(x)
        
        # 3. Feed-forward
        ff_output = self.ff_network(x)
        x = x + self.dropout(ff_output)
        x = self.norm3(x)
        
        return x


class TransformerDecoder(nn.Module):
    """
    Complete Transformer Decoder stack.
    
    Performs autoregressive generation with causal masking.
    Each position can only attend to earlier positions.
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
            vocab_size: Size of target vocabulary
            d_model: Dimension of model embeddings
            num_heads: Number of attention heads
            num_layers: Number of decoder layers
            d_ff: Dimension of feed-forward network
            max_seq_len: Maximum sequence length
            dropout: Dropout rate
            pad_token_id: ID of padding token
        """
        super().__init__()
        
        self.d_model = d_model
        self.pad_token_id = pad_token_id
        
        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        
        # Positional encoding (same as encoder)
        from encoder import PositionalEncoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Stack of decoder layers
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection to vocabulary
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # Final layer norm
        self.norm = nn.LayerNorm(d_model)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.output_projection.weight)
    
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        """
        Decode target sequence (used during training with teacher forcing).
        
        Args:
            tgt: Target token IDs [batch_size, tgt_len]
            memory: Encoder output [batch_size, src_len, d_model]
            tgt_mask: Causal mask [batch_size, tgt_len, tgt_len]
            memory_mask: Encoder padding mask [batch_size, src_len]
        
        Returns:
            Logits over vocabulary [batch_size, tgt_len, vocab_size]
        """
        # Create causal mask if not provided
        if tgt_mask is None:
            seq_len = tgt.size(1)
            tgt_mask = AttentionMaskGenerator.create_causal_mask(
                seq_len, device=tgt.device
            )
            # Remove batch and head dimensions for MultiheadAttention
            tgt_mask = tgt_mask.squeeze(0).squeeze(0)
        
        # Embed tokens and scale
        x = self.embedding(tgt) * math.sqrt(self.d_model)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Pass through decoder layers
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, memory_mask)
        
        # Final normalization
        x = self.norm(x)
        
        # Project to vocabulary
        logits = self.output_projection(x)
        
        return logits
    
    def generate_step(self, tgt, memory, memory_mask=None):
        """
        Generate next token logits (used during autoregressive inference).
        
        This is the KEY function for token-by-token generation.
        
        Args:
            tgt: Previously generated tokens [batch_size, current_len]
            memory: Encoder output [batch_size, src_len, d_model]
            memory_mask: Encoder padding mask
        
        Returns:
            Logits for next token [batch_size, vocab_size]
        """
        # Forward pass (causal mask created internally)
        logits = self.forward(tgt, memory, memory_mask=memory_mask)
        
        # Return logits for the last position (next token prediction)
        return logits[:, -1, :]  # [batch_size, vocab_size]


class AutoregressiveGenerator:
    """
    Handles autoregressive token-by-token generation.
    
    This is the CORE of the inference pipeline.
    Generates output one token at a time until end token or max length.
    """
    
    def __init__(self, decoder, max_len=100, bos_token_id=1, eos_token_id=2):
        """
        Args:
            decoder: TransformerDecoder instance
            max_len: Maximum generation length
            bos_token_id: Beginning of sequence token
            eos_token_id: End of sequence token
        """
        self.decoder = decoder
        self.max_len = max_len
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
    
    @torch.no_grad()
    def generate_greedy(self, memory, memory_mask=None):
        """
        Greedy autoregressive generation (pick most likely token each step).
        
        Generation loop:
            Step 0: input = [<BOS>], predict token_1
            Step 1: input = [<BOS>, token_1], predict token_2
            Step 2: input = [<BOS>, token_1, token_2], predict token_3
            ...
            
        Args:
            memory: Encoder output [batch_size, src_len, d_model]
            memory_mask: Encoder padding mask
        
        Returns:
            Generated token IDs [batch_size, gen_len]
        """
        batch_size = memory.size(0)
        device = memory.device
        
        # Start with BOS token
        generated = torch.full(
            (batch_size, 1),
            self.bos_token_id,
            dtype=torch.long,
            device=device
        )
        
        # Generate tokens one by one
        for step in range(self.max_len - 1):
            # Get next token logits
            logits = self.decoder.generate_step(generated, memory, memory_mask)
            
            # Greedy selection (argmax)
            next_token = logits.argmax(dim=-1, keepdim=True)  # [batch, 1]
            
            # Append to sequence
            generated = torch.cat([generated, next_token], dim=1)
            
            # Check if all sequences have generated EOS
            if (next_token == self.eos_token_id).all():
                break
        
        return generated
    
    @torch.no_grad()
    def generate_beam_search(self, memory, memory_mask=None, beam_width=4):
        """
        Beam search for better quality (keeps top-k hypotheses).
        
        Args:
            memory: Encoder output [batch_size, src_len, d_model]
            memory_mask: Encoder padding mask
            beam_width: Number of beams to keep
        
        Returns:
            Best generated sequence [batch_size, gen_len]
        """
        batch_size = memory.size(0)
        device = memory.device
        
        # Start with BOS token
        sequences = torch.full(
            (batch_size, beam_width, 1),
            self.bos_token_id,
            dtype=torch.long,
            device=device
        )
        
        # Track scores for each beam
        scores = torch.zeros(batch_size, beam_width, device=device)
        
        # Expand memory for beam search
        memory_expanded = memory.unsqueeze(1).expand(
            batch_size, beam_width, -1, -1
        ).reshape(batch_size * beam_width, memory.size(1), memory.size(2))
        
        for step in range(self.max_len - 1):
            # Reshape for decoder: [batch * beam, seq_len]
            current_seqs = sequences.reshape(batch_size * beam_width, -1)
            
            # Get logits
            logits = self.decoder.generate_step(current_seqs, memory_expanded)
            log_probs = torch.log_softmax(logits, dim=-1)
            
            # Reshape back: [batch, beam, vocab]
            log_probs = log_probs.reshape(batch_size, beam_width, -1)
            
            # Add to cumulative scores
            vocab_size = log_probs.size(-1)
            candidate_scores = scores.unsqueeze(-1) + log_probs
            
            # Flatten and get top beam_width
            candidate_scores = candidate_scores.reshape(batch_size, -1)
            top_scores, top_indices = candidate_scores.topk(beam_width, dim=-1)
            
            # Convert flat indices back to (beam, token) indices
            beam_indices = top_indices // vocab_size
            token_indices = top_indices % vocab_size
            
            # Update sequences and scores
            new_sequences = []
            for b in range(batch_size):
                new_seqs = []
                for k in range(beam_width):
                    beam_idx = beam_indices[b, k]
                    token_idx = token_indices[b, k]
                    
                    prev_seq = sequences[b, beam_idx]
                    new_seq = torch.cat([prev_seq, token_idx.unsqueeze(0)])
                    new_seqs.append(new_seq)
                
                new_sequences.append(torch.stack(new_seqs))
            
            sequences = torch.stack(new_sequences)
            scores = top_scores
            
            # Early stopping if all beams end with EOS
            if (sequences[:, :, -1] == self.eos_token_id).all():
                break
        
        # Return best sequence (highest score) for each batch
        best_beam_idx = scores.argmax(dim=1)
        best_sequences = sequences[range(batch_size), best_beam_idx]
        
        return best_sequences


# Testing and demonstration
if __name__ == "__main__":
    print("=" * 70)
    print("DECODER MODULE DEMONSTRATION")
    print("=" * 70)
    
    # Configuration
    vocab_size = 10000
    d_model = 512
    batch_size = 2
    src_len = 20
    tgt_len = 15
    
    print(f"\nConfiguration:")
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Model dimension: {d_model}")
    print(f"  Source length: {src_len}")
    print(f"  Target length: {tgt_len}")
    
    # Create decoder
    decoder = TransformerDecoder(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=8,
        num_layers=6
    )
    
    print(f"\n✓ Decoder created with {sum(p.numel() for p in decoder.parameters()):,} parameters")
    
    # Create sample inputs
    memory = torch.randn(batch_size, src_len, d_model)  # From encoder
    tgt = torch.randint(1, vocab_size, (batch_size, tgt_len))  # Target tokens
    
    print(f"\nMemory (encoder output) shape: {memory.shape}")
    print(f"Target tokens shape: {tgt.shape}")
    
    # Forward pass (training mode)
    decoder.eval()
    with torch.no_grad():
        logits = decoder(tgt, memory)
    
    print(f"\nDecoder output (logits) shape: {logits.shape}")
    print(f"  → Predicts next token for each of {tgt_len} positions")
    
    # Test autoregressive generation
    print("\n" + "-" * 70)
    print("AUTOREGRESSIVE GENERATION TEST")
    print("-" * 70)
    
    generator = AutoregressiveGenerator(decoder, max_len=20)
    generated = generator.generate_greedy(memory)
    
    print(f"\nGenerated sequence shape: {generated.shape}")
    print(f"Generated tokens (first sequence): {generated[0].tolist()}")
    
    print("\n" + "=" * 70)
    print("✓ Decoder and autoregressive generation working correctly!")
    print("=" * 70)
