"""
Complete Transformer Seq2Seq Model

Combines encoder and decoder into a unified architecture for sequence-to-sequence tasks.
Primary use case: Text Summarization

Architecture Flow:
    1. Input Document → Encoder → Memory (contextual representations)
    2. Memory + <BOS> → Decoder → Summary Token 1
    3. Memory + [<BOS>, Token1] → Decoder → Summary Token 2
    4. Memory + [<BOS>, Token1, Token2] → Decoder → Summary Token 3
    ... (autoregressive generation continues)

This module provides both:
- Training interface (teacher forcing)
- Inference interface (autoregressive generation)
"""

import torch
import torch.nn as nn
from encoder import TransformerEncoder, PretrainedEncoderWrapper
from decoder import TransformerDecoder, AutoregressiveGenerator
from attention_masks import AttentionMaskGenerator


class Seq2SeqTransformer(nn.Module):
    """
    Complete Transformer Encoder-Decoder for sequence-to-sequence tasks.
    
    Key Features:
    - Modular encoder/decoder architecture
    - Support for both custom and pre-trained components
    - Teacher forcing during training
    - Autoregressive generation during inference
    """
    
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=512,
        num_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        d_ff=2048,
        max_seq_len=512,
        dropout=0.1,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2
    ):
        """
        Args:
            src_vocab_size: Source vocabulary size
            tgt_vocab_size: Target vocabulary size
            d_model: Model dimension
            num_heads: Number of attention heads
            num_encoder_layers: Number of encoder layers
            num_decoder_layers: Number of decoder layers
            d_ff: Feed-forward dimension
            max_seq_len: Maximum sequence length
            dropout: Dropout rate
            pad_token_id: Padding token ID
            bos_token_id: Beginning of sequence token ID
            eos_token_id: End of sequence token ID
        """
        super().__init__()
        
        self.d_model = d_model
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        
        # Encoder
        self.encoder = TransformerEncoder(
            vocab_size=src_vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_encoder_layers,
            d_ff=d_ff,
            max_seq_len=max_seq_len,
            dropout=dropout,
            pad_token_id=pad_token_id
        )
        
        # Decoder
        self.decoder = TransformerDecoder(
            vocab_size=tgt_vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_decoder_layers,
            d_ff=d_ff,
            max_seq_len=max_seq_len,
            dropout=dropout,
            pad_token_id=pad_token_id
        )
        
        # Autoregressive generator
        self.generator = AutoregressiveGenerator(
            decoder=self.decoder,
            max_len=max_seq_len,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id
        )
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        Forward pass for training (with teacher forcing).
        
        Teacher forcing: Feed ground-truth tokens to decoder during training,
        rather than previously generated tokens. This speeds up training.
        
        Args:
            src: Source sequence [batch_size, src_len]
            tgt: Target sequence [batch_size, tgt_len]
            src_mask: Source padding mask
            tgt_mask: Target causal mask
        
        Returns:
            Logits [batch_size, tgt_len, vocab_size]
        """
        # Encode source
        memory = self.encoder(src, src_mask)
        
        # Create memory mask (encoder padding mask)
        if src_mask is None:
            memory_mask = (src == self.pad_token_id)
        else:
            memory_mask = src_mask.squeeze(1).squeeze(1).bool()
        
        # Decode target (with teacher forcing)
        output = self.decoder(tgt, memory, tgt_mask, memory_mask)
        
        return output
    
    def encode(self, src, src_mask=None):
        """
        Encode source sequence.
        
        Args:
            src: Source tokens [batch_size, src_len]
            src_mask: Optional padding mask
        
        Returns:
            Memory (encoder output) [batch_size, src_len, d_model]
        """
        return self.encoder(src, src_mask)
    
    def decode(self, tgt, memory, tgt_mask=None, memory_mask=None):
        """
        Decode target sequence given encoder memory.
        
        Args:
            tgt: Target tokens [batch_size, tgt_len]
            memory: Encoder output [batch_size, src_len, d_model]
            tgt_mask: Causal mask
            memory_mask: Encoder padding mask
        
        Returns:
            Logits [batch_size, tgt_len, vocab_size]
        """
        return self.decoder(tgt, memory, tgt_mask, memory_mask)
    
    @torch.no_grad()
    def generate(self, src, src_mask=None, method='greedy', **kwargs):
        """
        Generate output sequence autoregressively.
        
        This is the main inference function for summarization.
        
        Args:
            src: Source tokens [batch_size, src_len]
            src_mask: Optional padding mask
            method: 'greedy' or 'beam_search'
            **kwargs: Additional arguments for generation (e.g., beam_width)
        
        Returns:
            Generated sequence [batch_size, gen_len]
        """
        # Encode source
        memory = self.encoder(src, src_mask)
        
        # Create memory mask
        if src_mask is None:
            memory_mask = (src == self.pad_token_id)
        else:
            memory_mask = src_mask.squeeze(1).squeeze(1).bool()
        
        # Generate
        if method == 'greedy':
            return self.generator.generate_greedy(memory, memory_mask)
        elif method == 'beam_search':
            beam_width = kwargs.get('beam_width', 4)
            return self.generator.generate_beam_search(memory, memory_mask, beam_width)
        else:
            raise ValueError(f"Unknown generation method: {method}")


class PretrainedSeq2SeqWrapper(nn.Module):
    """
    Wrapper for pre-trained Seq2Seq models (T5, BART, etc.)
    
    Allows using state-of-the-art pre-trained models while maintaining
    our custom interface and architecture understanding.
    """
    
    def __init__(self, model_name="t5-small"):
        super().__init__()
        
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            
            # Load model and tokenizer
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Extract special token IDs
            self.pad_token_id = self.tokenizer.pad_token_id
            self.bos_token_id = self.tokenizer.bos_token_id or self.tokenizer.pad_token_id
            self.eos_token_id = self.tokenizer.eos_token_id
            
            print(f"✓ Loaded pre-trained model: {model_name}")
            print(f"  Encoder layers: {self.model.config.num_layers}")
            print(f"  Decoder layers: {self.model.config.num_decoder_layers}")
            print(f"  Model dimension: {self.model.config.d_model}")
            
        except ImportError:
            raise ImportError(
                "transformers library required. Install with: pip install transformers"
            )
    
    def forward(self, input_ids, decoder_input_ids, attention_mask=None):
        """
        Forward pass through pre-trained model.
        
        Args:
            input_ids: Source tokens
            decoder_input_ids: Target tokens
            attention_mask: Source attention mask
        
        Returns:
            Model output with logits
        """
        return self.model(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            attention_mask=attention_mask
        )
    
    @torch.no_grad()
    def generate(self, input_ids, attention_mask=None, max_length=100, num_beams=4):
        """
        Generate summary using pre-trained model.
        
        Args:
            input_ids: Source tokens
            attention_mask: Source mask
            max_length: Maximum generation length
            num_beams: Beam width for beam search
        
        Returns:
            Generated token IDs
        """
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True
        )
    
    def encode_text(self, text):
        """
        Tokenize input text.
        
        Args:
            text: Input string or list of strings
        
        Returns:
            Tokenized inputs
        """
        return self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
    
    def decode_tokens(self, token_ids, skip_special_tokens=True):
        """
        Convert token IDs back to text.
        
        Args:
            token_ids: Generated token IDs
            skip_special_tokens: Whether to remove special tokens
        
        Returns:
            Decoded text
        """
        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens
        )


# Testing and demonstration
if __name__ == "__main__":
    print("=" * 70)
    print("COMPLETE TRANSFORMER SEQ2SEQ MODEL DEMONSTRATION")
    print("=" * 70)
    
    # Configuration
    src_vocab_size = 10000
    tgt_vocab_size = 8000
    d_model = 256  # Smaller for demo
    batch_size = 2
    src_len = 30
    tgt_len = 20
    
    print(f"\nConfiguration:")
    print(f"  Source vocab: {src_vocab_size}")
    print(f"  Target vocab: {tgt_vocab_size}")
    print(f"  Model dimension: {d_model}")
    print(f"  Encoder layers: 4")
    print(f"  Decoder layers: 4")
    
    # Create model
    model = Seq2SeqTransformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        num_heads=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
        d_ff=1024
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n✓ Model created with {total_params:,} parameters")
    
    # Create sample inputs
    src = torch.randint(1, src_vocab_size, (batch_size, src_len))
    tgt = torch.randint(1, tgt_vocab_size, (batch_size, tgt_len))
    
    # Add some padding
    src[:, -5:] = 0
    tgt[:, -3:] = 0
    
    print(f"\nInput shapes:")
    print(f"  Source: {src.shape}")
    print(f"  Target: {tgt.shape}")
    
    # Test forward pass (training)
    print("\n" + "-" * 70)
    print("TESTING FORWARD PASS (Training Mode)")
    print("-" * 70)
    
    model.eval()
    with torch.no_grad():
        logits = model(src, tgt)
    
    print(f"Output logits shape: {logits.shape}")
    print(f"  → Predicts distribution over {tgt_vocab_size} tokens")
    print(f"  → For each of {tgt_len} positions")
    
    # Test generation (inference)
    print("\n" + "-" * 70)
    print("TESTING GENERATION (Inference Mode)")
    print("-" * 70)
    
    generated_greedy = model.generate(src, method='greedy')
    print(f"\nGreedy generation shape: {generated_greedy.shape}")
    print(f"Generated sequence (first sample):")
    print(f"  {generated_greedy[0].tolist()[:15]}... (truncated)")
    
    # Test beam search
    generated_beam = model.generate(src, method='beam_search', beam_width=4)
    print(f"\nBeam search generation shape: {generated_beam.shape}")
    print(f"Generated sequence (first sample):")
    print(f"  {generated_beam[0].tolist()[:15]}... (truncated)")
    
    print("\n" + "=" * 70)
    print("✓ Complete Seq2Seq Transformer working correctly!")
    print("=" * 70)
    
    # Show architecture summary
    print("\n" + "=" * 70)
    print("ARCHITECTURE SUMMARY")
    print("=" * 70)
    print(f"""
    INPUT DOCUMENT (tokens)
           ↓
    ┌──────────────────┐
    │  ENCODER         │  ← Bidirectional attention
    │  {model.encoder.layers.__len__()} layers          │  ← Reads entire input
    └──────────────────┘
           ↓
        MEMORY (contextual representations)
           ↓
    ┌──────────────────┐
    │  DECODER         │  ← Causal (autoregressive) attention
    │  {model.decoder.layers.__len__()} layers          │  ← Generates one token at a time
    └──────────────────┘     ← Also cross-attends to encoder memory
           ↓
    OUTPUT SUMMARY (tokens)
    
    Key Mechanisms:
    ✓ Encoder: Bidirectional self-attention
    ✓ Decoder: Causal self-attention (masked)
    ✓ Cross-attention: Decoder → Encoder memory
    ✓ Autoregressive: Generate token-by-token
    """)
