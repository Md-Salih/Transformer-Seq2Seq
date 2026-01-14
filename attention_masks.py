"""
Attention Mask Generation for Transformer Seq2Seq

This module handles three types of attention masks:
1. Padding masks - Hide padding tokens from attention
2. Causal (look-ahead) masks - Prevent decoder from seeing future tokens
3. Combined masks - Apply both padding and causal constraints

Key Concept: Causal Masking
----------------------------
In autoregressive generation, the decoder must only attend to previously
generated tokens, not future ones. This is enforced by a triangular mask.

Example (4 tokens):
    t0  t1  t2  t3
t0 [ T   F   F   F ]  ← Token 0 can only see itself
t1 [ T   T   F   F ]  ← Token 1 can see tokens 0-1
t2 [ T   T   T   F ]  ← Token 2 can see tokens 0-2
t3 [ T   T   T   T ]  ← Token 3 can see tokens 0-3

This ensures each position can only depend on earlier positions during training.
"""

import torch
import torch.nn as nn


class AttentionMaskGenerator:
    """
    Production-grade attention mask generator for Transformer models.
    Supports both encoder (bidirectional) and decoder (causal) masks.
    """
    
    @staticmethod
    def create_padding_mask(seq, pad_token_id=0):
        """
        Create padding mask to ignore pad tokens in attention.
        
        Args:
            seq (torch.Tensor): Input sequence [batch_size, seq_len]
            pad_token_id (int): ID of padding token (default: 0)
            
        Returns:
            torch.Tensor: Boolean mask [batch_size, 1, 1, seq_len]
                         True where padding exists, False otherwise
        """
        # Create mask where padding tokens are True (to be masked out)
        mask = (seq == pad_token_id)
        
        # Add dimensions for broadcasting: [batch, 1, 1, seq_len]
        # This allows the mask to broadcast across all attention heads and query positions
        return mask.unsqueeze(1).unsqueeze(2)
    
    @staticmethod
    def create_causal_mask(seq_len, device='cpu'):
        """
        Create causal (look-ahead) mask for autoregressive decoding.
        
        This is THE KEY to autoregressive generation. Each position can only
        attend to itself and previous positions, preventing information leakage
        from future tokens during training.
        
        Args:
            seq_len (int): Length of the sequence
            device (str or torch.device): Device to create mask on
            
        Returns:
            torch.Tensor: Boolean mask [1, 1, seq_len, seq_len]
                         True where attention should be masked (future positions)
        
        Example for seq_len=4:
            [[[[False,  True,  True,  True],
               [False, False,  True,  True],
               [False, False, False,  True],
               [False, False, False, False]]]]
        """
        # Create upper triangular matrix (excluding diagonal)
        # torch.triu with diagonal=1 creates upper triangle above main diagonal
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        
        # Convert to boolean (1 → True, 0 → False)
        mask = mask.bool()
        
        # Add batch and head dimensions: [1, 1, seq_len, seq_len]
        return mask.unsqueeze(0).unsqueeze(0)
    
    @staticmethod
    def create_combined_mask(target_seq, pad_token_id=0):
        """
        Create combined padding + causal mask for decoder self-attention.
        
        During training, the decoder needs both:
        1. Causal mask: Don't look at future tokens
        2. Padding mask: Don't attend to padding tokens
        
        Args:
            target_seq (torch.Tensor): Target sequence [batch_size, seq_len]
            pad_token_id (int): Padding token ID
            
        Returns:
            torch.Tensor: Combined boolean mask [batch_size, 1, seq_len, seq_len]
        """
        batch_size, seq_len = target_seq.shape
        device = target_seq.device
        
        # Create causal mask
        causal_mask = AttentionMaskGenerator.create_causal_mask(seq_len, device)
        
        # Create padding mask
        padding_mask = AttentionMaskGenerator.create_padding_mask(target_seq, pad_token_id)
        
        # Expand padding mask to [batch, 1, seq_len, seq_len]
        # Each query position should ignore padding in all key positions
        padding_mask = padding_mask.expand(batch_size, 1, seq_len, seq_len)
        
        # Combine: mask if EITHER causal OR padding requires it
        combined_mask = causal_mask | padding_mask
        
        return combined_mask
    
    @staticmethod
    def visualize_causal_mask(seq_len=8):
        """
        Utility function to visualize causal mask structure.
        Useful for debugging and understanding.
        
        Args:
            seq_len (int): Sequence length to visualize
        """
        mask = AttentionMaskGenerator.create_causal_mask(seq_len)
        mask_2d = mask.squeeze().int()  # Remove batch/head dims, convert to int
        
        print(f"\n{'='*50}")
        print(f"CAUSAL MASK VISUALIZATION (seq_len={seq_len})")
        print(f"{'='*50}")
        print("1 = MASKED (cannot attend), 0 = VISIBLE (can attend)\n")
        print("        " + "  ".join([f"t{i}" for i in range(seq_len)]))
        print("      " + "-" * (4 * seq_len))
        
        for i in range(seq_len):
            row = mask_2d[i].tolist()
            row_str = "  ".join([str(x) for x in row])
            print(f"  t{i}  |  {row_str}")
        
        print(f"\n{'='*50}\n")


class AttentionMaskConverter:
    """
    Convert between different mask representations for various libraries.
    """
    
    @staticmethod
    def to_additive_mask(boolean_mask):
        """
        Convert boolean mask to additive mask for scaled dot-product attention.
        
        Attention computation: softmax((Q @ K.T) / sqrt(d) + mask)
        - Masked positions get -inf, which become 0 after softmax
        - Unmasked positions get 0, which don't affect the scores
        
        Args:
            boolean_mask (torch.Tensor): Boolean mask (True = mask, False = keep)
            
        Returns:
            torch.Tensor: Additive mask (masked = -inf, unmasked = 0)
        """
        additive_mask = torch.zeros_like(boolean_mask, dtype=torch.float32)
        additive_mask.masked_fill_(boolean_mask, float('-inf'))
        return additive_mask
    
    @staticmethod
    def to_binary_mask(boolean_mask):
        """
        Convert boolean mask to binary (0/1) mask.
        
        Args:
            boolean_mask (torch.Tensor): Boolean mask
            
        Returns:
            torch.Tensor: Binary mask (1 = mask, 0 = keep)
        """
        return boolean_mask.long()


# Example usage and testing
if __name__ == "__main__":
    print("=" * 70)
    print("ATTENTION MASKS DEMONSTRATION")
    print("=" * 70)
    
    # Example 1: Visualize causal mask
    print("\n[1] CAUSAL MASK (prevents looking at future tokens)")
    AttentionMaskGenerator.visualize_causal_mask(seq_len=6)
    
    # Example 2: Padding mask
    print("\n[2] PADDING MASK (ignores padding tokens)")
    sample_seq = torch.tensor([[1, 2, 3, 4, 0, 0],  # Last 2 are padding
                               [1, 2, 0, 0, 0, 0]])  # Last 4 are padding
    
    padding_mask = AttentionMaskGenerator.create_padding_mask(sample_seq, pad_token_id=0)
    print(f"Input sequence:\n{sample_seq}")
    print(f"\nPadding mask (True = pad, False = real token):")
    print(padding_mask.squeeze())
    
    # Example 3: Combined mask
    print("\n[3] COMBINED MASK (causal + padding)")
    combined = AttentionMaskGenerator.create_combined_mask(sample_seq, pad_token_id=0)
    print(f"Combined mask for first sequence (causal + padding):")
    print(combined[0, 0].int())
    
    # Example 4: Additive mask conversion
    print("\n[4] ADDITIVE MASK (for attention computation)")
    causal = AttentionMaskGenerator.create_causal_mask(4)
    additive = AttentionMaskConverter.to_additive_mask(causal)
    print("Causal mask (boolean):")
    print(causal.squeeze().int())
    print("\nAdditive mask (for softmax):")
    print(additive.squeeze())
    
    print("\n" + "=" * 70)
    print("✓ All mask types demonstrated successfully!")
    print("=" * 70)
