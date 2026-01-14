"""
Training Pipeline for Transformer Seq2Seq Summarization

This module handles:
1. Data loading and preprocessing
2. Teacher forcing training
3. Loss calculation and optimization
4. Model checkpointing
5. Training metrics and logging

Key Training Concept: TEACHER FORCING
-------------------------------------
During training, we feed ground-truth tokens to the decoder rather than
previously generated tokens. This speeds up training significantly.

Example:
    Target: "The cat sat"
    Step 1: Input=[<BOS>], Predict="The"
    Step 2: Input=[<BOS>, "The"], Predict="cat"  ← Uses ground-truth "The"
    Step 3: Input=[<BOS>, "The", "cat"], Predict="sat"  ← Uses ground-truth tokens
    
During inference, we use actual predictions (autoregressive generation).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformer import Seq2SeqTransformer, PretrainedSeq2SeqWrapper
from attention_masks import AttentionMaskGenerator
import os
import json
from tqdm import tqdm
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SummarizationDataset(Dataset):
    """
    Dataset for text summarization.
    
    Expects data in format:
    [
        {"document": "Long text...", "summary": "Short summary..."},
        ...
    ]
    """
    
    def __init__(self, data, tokenizer, max_src_len=512, max_tgt_len=128):
        """
        Args:
            data: List of {"document": ..., "summary": ...} dicts
            tokenizer: Tokenizer instance
            max_src_len: Maximum source sequence length
            max_tgt_len: Maximum target sequence length
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenize source and target
        src = self.tokenizer.encode_text(
            item['document'],
            max_length=self.max_src_len
        )
        
        tgt = self.tokenizer.encode_text(
            item['summary'],
            max_length=self.max_tgt_len
        )
        
        return {
            'src_ids': src['input_ids'].squeeze(0),
            'src_mask': src['attention_mask'].squeeze(0),
            'tgt_ids': tgt['input_ids'].squeeze(0),
            'tgt_mask': tgt['attention_mask'].squeeze(0)
        }


class SimpleTokenizer:
    """
    Simple character-level tokenizer for demonstration.
    For production, use transformers.AutoTokenizer.
    """
    
    def __init__(self, vocab=None):
        if vocab is None:
            # Create simple vocab
            vocab = ['<PAD>', '<BOS>', '<EOS>', '<UNK>']
            vocab += [chr(i) for i in range(32, 127)]  # Printable ASCII
        
        self.vocab = vocab
        self.token2id = {token: idx for idx, token in enumerate(vocab)}
        self.id2token = {idx: token for token, idx in self.token2id.items()}
        
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.unk_token_id = 3
    
    def encode_text(self, text, max_length=512):
        """Encode text to token IDs."""
        if isinstance(text, list):
            text = text[0]  # Handle batch input
        
        # Convert to IDs
        ids = [self.bos_token_id]
        for char in text[:max_length - 2]:
            ids.append(self.token2id.get(char, self.unk_token_id))
        ids.append(self.eos_token_id)
        
        # Pad to max_length
        attention_mask = [1] * len(ids)
        while len(ids) < max_length:
            ids.append(self.pad_token_id)
            attention_mask.append(0)
        
        return {
            'input_ids': torch.tensor([ids]),
            'attention_mask': torch.tensor([attention_mask])
        }
    
    def decode_tokens(self, token_ids, skip_special_tokens=True):
        """Decode token IDs to text."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        
        chars = []
        for tid in token_ids:
            if skip_special_tokens and tid in [self.pad_token_id, self.bos_token_id, self.eos_token_id]:
                continue
            chars.append(self.id2token.get(tid, '<UNK>'))
        
        return ''.join(chars)


class Trainer:
    """
    Training pipeline for Seq2Seq Transformer.
    
    Handles:
    - Training loop with teacher forcing
    - Validation
    - Checkpointing
    - Metrics logging
    """
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        device='cuda',
        checkpoint_dir='checkpoints'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch in pbar:
            # Move to device
            src_ids = batch['src_ids'].to(self.device)
            tgt_ids = batch['tgt_ids'].to(self.device)
            
            # Prepare decoder input (shift right)
            # Input: [<BOS>, token1, token2, ...]
            # Target: [token1, token2, ..., <EOS>]
            decoder_input = tgt_ids[:, :-1]
            targets = tgt_ids[:, 1:]
            
            # Forward pass with teacher forcing
            self.optimizer.zero_grad()
            logits = self.model(src_ids, decoder_input)
            
            # Calculate loss
            loss = self.criterion(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1)
            )
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    @torch.no_grad()
    def validate(self):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        
        for batch in tqdm(self.val_loader, desc='Validation'):
            src_ids = batch['src_ids'].to(self.device)
            tgt_ids = batch['tgt_ids'].to(self.device)
            
            decoder_input = tgt_ids[:, :-1]
            targets = tgt_ids[:, 1:]
            
            logits = self.model(src_ids, decoder_input)
            
            loss = self.criterion(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1)
            )
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        return avg_loss
    
    def save_checkpoint(self, epoch, val_loss, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        
        path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path}")
        
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            logger.info(f"Best model saved: {best_path}")
    
    def train(self, num_epochs, save_every=5):
        """
        Complete training loop.
        
        Args:
            num_epochs: Number of epochs to train
            save_every: Save checkpoint every N epochs
        """
        best_val_loss = float('inf')
        
        logger.info("=" * 70)
        logger.info("STARTING TRAINING")
        logger.info("=" * 70)
        
        for epoch in range(1, num_epochs + 1):
            logger.info(f"\nEpoch {epoch}/{num_epochs}")
            logger.info("-" * 70)
            
            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate()
            self.val_losses.append(val_loss)
            
            logger.info(f"Train Loss: {train_loss:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f}")
            
            # Save checkpoint
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                logger.info("✓ New best model!")
            
            if epoch % save_every == 0 or is_best:
                self.save_checkpoint(epoch, val_loss, is_best)
        
        logger.info("\n" + "=" * 70)
        logger.info("TRAINING COMPLETE")
        logger.info(f"Best validation loss: {best_val_loss:.4f}")
        logger.info("=" * 70)


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("TRAINING PIPELINE DEMONSTRATION")
    print("=" * 70)
    
    # Create simple dataset
    sample_data = [
        {
            "document": "The quick brown fox jumps over the lazy dog. It was a sunny day.",
            "summary": "Fox jumps over dog."
        },
        {
            "document": "Machine learning is transforming industries worldwide. AI is the future.",
            "summary": "ML transforms industries."
        },
        {
            "document": "Climate change poses significant challenges. We must act now.",
            "summary": "Climate action needed."
        }
    ] * 10  # Repeat for demo
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    # For demo, use simple tokenizer
    tokenizer = SimpleTokenizer()
    
    # Create datasets
    train_data = sample_data[:25]
    val_data = sample_data[25:]
    
    # For production with pre-trained models:
    print("\n" + "-" * 70)
    print("USING PRE-TRAINED MODEL (Recommended for Production)")
    print("-" * 70)
    
    try:
        # Use pre-trained T5 model
        model = PretrainedSeq2SeqWrapper("t5-small")
        
        print("\n✓ Pre-trained model loaded successfully")
        print("  This model is production-ready and fine-tunable!")
        
    except Exception as e:
        print(f"\nNote: Pre-trained model requires 'transformers' library")
        print(f"Install with: pip install transformers")
        print(f"\nUsing custom model for demo instead...")
        
        # Use custom model
        model = Seq2SeqTransformer(
            src_vocab_size=len(tokenizer.vocab),
            tgt_vocab_size=len(tokenizer.vocab),
            d_model=256,
            num_heads=8,
            num_encoder_layers=3,
            num_decoder_layers=3,
            d_ff=1024
        )
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\n" + "=" * 70)
    print("TRAINING PIPELINE READY")
    print("=" * 70)
    print("""
    To train the model:
    
    1. Prepare your dataset (documents + summaries)
    2. Create DataLoader with your data
    3. Initialize optimizer and loss function
    4. Create Trainer instance
    5. Call trainer.train(num_epochs)
    
    Example:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        trainer = Trainer(model, train_loader, val_loader, optimizer, criterion)
        trainer.train(num_epochs=10)
    """)
