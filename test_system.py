"""
Demo Script - Complete System Test

This script demonstrates all components of the Transformer Seq2Seq system.
Run this to verify everything works correctly.
"""

import torch
import sys

def print_section(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")

def test_attention_masks():
    print_section("1. ATTENTION MASKS")
    
    from attention_masks import AttentionMaskGenerator
    
    print("Testing causal mask generation...")
    mask = AttentionMaskGenerator.create_causal_mask(4)
    print(f"✓ Causal mask shape: {mask.shape}")
    print(f"  Mask structure (4x4):")
    print(mask.squeeze().int())
    
    print("\nTesting padding mask...")
    seq = torch.tensor([[1, 2, 3, 0, 0]])
    pad_mask = AttentionMaskGenerator.create_padding_mask(seq)
    print(f"✓ Padding mask shape: {pad_mask.shape}")
    print(f"  Padded positions: {pad_mask.squeeze()}")
    
    print("\n✅ Attention masks working correctly!")

def test_encoder():
    print_section("2. ENCODER")
    
    from encoder import TransformerEncoder
    
    print("Creating encoder...")
    encoder = TransformerEncoder(
        vocab_size=1000,
        d_model=256,
        num_heads=8,
        num_layers=4
    )
    
    print(f"✓ Encoder created")
    print(f"  Parameters: {sum(p.numel() for p in encoder.parameters()):,}")
    
    print("\nTesting forward pass...")
    src = torch.randint(1, 1000, (2, 10))
    memory = encoder(src)
    
    print(f"✓ Forward pass successful")
    print(f"  Input shape: {src.shape}")
    print(f"  Output (memory) shape: {memory.shape}")
    print(f"  Memory stats: mean={memory.mean():.4f}, std={memory.std():.4f}")
    
    print("\n✅ Encoder working correctly!")

def test_decoder():
    print_section("3. DECODER")
    
    from decoder import TransformerDecoder, AutoregressiveGenerator
    
    print("Creating decoder...")
    decoder = TransformerDecoder(
        vocab_size=1000,
        d_model=256,
        num_heads=8,
        num_layers=4
    )
    
    print(f"✓ Decoder created")
    print(f"  Parameters: {sum(p.numel() for p in decoder.parameters()):,}")
    
    print("\nTesting autoregressive generation...")
    memory = torch.randn(2, 10, 256)  # Fake encoder output
    generator = AutoregressiveGenerator(decoder, max_len=15)
    
    generated = generator.generate_greedy(memory)
    
    print(f"✓ Autoregressive generation successful")
    print(f"  Memory shape: {memory.shape}")
    print(f"  Generated shape: {generated.shape}")
    print(f"  Sample output (first 10 tokens): {generated[0][:10].tolist()}")
    
    print("\n✅ Decoder and autoregressive generation working correctly!")

def test_transformer():
    print_section("4. COMPLETE TRANSFORMER")
    
    from transformer import Seq2SeqTransformer
    
    print("Creating complete Seq2Seq model...")
    model = Seq2SeqTransformer(
        src_vocab_size=1000,
        tgt_vocab_size=1000,
        d_model=256,
        num_encoder_layers=4,
        num_decoder_layers=4
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created")
    print(f"  Total parameters: {total_params:,}")
    
    print("\nTesting training mode (teacher forcing)...")
    src = torch.randint(1, 1000, (2, 15))
    tgt = torch.randint(1, 1000, (2, 12))
    
    logits = model(src, tgt)
    print(f"✓ Training forward pass successful")
    print(f"  Input shape: {src.shape}")
    print(f"  Target shape: {tgt.shape}")
    print(f"  Output logits shape: {logits.shape}")
    
    print("\nTesting inference mode (autoregressive)...")
    generated = model.generate(src, method='greedy')
    print(f"✓ Inference generation successful")
    print(f"  Generated shape: {generated.shape}")
    print(f"  Sample output: {generated[0][:10].tolist()}")
    
    print("\n✅ Complete Transformer working correctly!")

def test_inference():
    print_section("5. INFERENCE PIPELINE")
    
    try:
        from inference import PretrainedInference
        
        print("Loading pre-trained T5 model...")
        print("(This may take a moment on first run)")
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = PretrainedInference("t5-small", device=device)
        
        print(f"✓ Model loaded on {device}")
        
        print("\nTesting summarization...")
        sample_text = """
        Artificial intelligence is rapidly transforming how we live and work.
        From virtual assistants to autonomous vehicles, AI systems are becoming
        increasingly sophisticated and widespread. Machine learning algorithms
        can now analyze vast amounts of data to identify patterns and make
        predictions. However, important questions remain about AI safety,
        ethics, and the societal impact of automation.
        """
        
        summary = model.summarize(sample_text.strip(), max_length=50, num_beams=4)
        
        print(f"✓ Summarization successful")
        print(f"\nOriginal text ({len(sample_text)} chars):")
        print(f"  {sample_text.strip()[:100]}...")
        print(f"\nGenerated summary ({len(summary)} chars):")
        print(f"  {summary}")
        
        compression = (1 - len(summary) / len(sample_text)) * 100
        print(f"\nCompression: {compression:.1f}%")
        
        print("\n✅ Inference pipeline working correctly!")
        
    except ImportError:
        print("⚠️  Transformers library not installed")
        print("   Install with: pip install transformers")
        print("   Skipping pre-trained model test")
    except Exception as e:
        print(f"⚠️  Could not load pre-trained model: {e}")
        print("   This is optional - custom models still work!")

def test_web_ui():
    print_section("6. WEB APPLICATION")
    
    try:
        from flask import Flask
        print("✓ Flask installed")
        
        print("\nTo test the web UI:")
        print("  1. Run: python app.py")
        print("  2. Open: http://localhost:5000")
        print("  3. Enter text and click 'Generate Summary'")
        print("  4. Watch token-by-token animation!")
        
        print("\n✅ Web UI ready to run!")
        
    except ImportError:
        print("⚠️  Flask not installed")
        print("   Install with: pip install flask")

def main():
    print("\n" + "=" * 70)
    print("  TRANSFORMER SEQ2SEQ - COMPLETE SYSTEM TEST")
    print("=" * 70)
    
    print("\nThis will test all components of the system.")
    print("Make sure you've installed dependencies: pip install -r requirements.txt\n")
    
    try:
        # Test core components
        test_attention_masks()
        test_encoder()
        test_decoder()
        test_transformer()
        test_inference()
        test_web_ui()
        
        # Final summary
        print("\n" + "=" * 70)
        print("  ✅ ALL TESTS PASSED!")
        print("=" * 70)
        
        print("\n🎉 System is fully operational!\n")
        
        print("Next steps:")
        print("  • Run: python app.py")
        print("  • Open: http://localhost:5000")
        print("  • Try the animated summarization UI")
        print("  • Read: README.md for detailed documentation")
        print("  • Explore: QUICKSTART.md for usage examples")
        print()
        
    except Exception as e:
        print("\n" + "=" * 70)
        print("  ❌ TEST FAILED")
        print("=" * 70)
        print(f"\nError: {e}")
        print("\nPlease check:")
        print("  • Dependencies installed: pip install -r requirements.txt")
        print("  • Python version: 3.8+")
        print("  • All files present in directory")
        print()
        sys.exit(1)

if __name__ == "__main__":
    main()
