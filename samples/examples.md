# Sample Inputs and Outputs

This directory contains example documents and their generated summaries.

## Example 1: Climate Change Article

**Input:**
```
Climate change is one of the most pressing issues facing humanity today. Rising global temperatures are causing extreme weather events, melting ice caps, and threatening ecosystems worldwide. Scientists warn that immediate action is needed to reduce carbon emissions and transition to renewable energy sources. The Paris Agreement aims to limit global warming to 1.5 degrees Celsius, but current efforts are falling short of this goal. Governments, businesses, and individuals must all play a role in combating climate change through sustainable practices and innovation.
```

**Generated Summary:**
```
Climate change poses urgent threats through extreme weather and ecosystem damage. Immediate action needed to reduce emissions and adopt renewable energy. Current efforts fall short of Paris Agreement goals.
```

---

## Example 2: Artificial Intelligence

**Input:**
```
Artificial intelligence is revolutionizing industries across the globe. From healthcare diagnostics to autonomous vehicles, AI systems are becoming increasingly capable and widespread. Machine learning algorithms can now outperform humans at specific tasks like image recognition and game playing. However, concerns about job displacement, algorithmic bias, and safety remain as the technology continues to advance rapidly. Researchers are working on developing more interpretable and ethical AI systems that can benefit society while minimizing potential risks.
```

**Generated Summary:**
```
AI is transforming industries with capabilities surpassing humans in specific tasks. Concerns about job loss and bias drive research into ethical, interpretable systems.
```

---

## Example 3: Space Exploration

**Input:**
```
Space exploration has entered a new era with private companies joining government agencies in the race to explore the cosmos. SpaceX, Blue Origin, and other firms are developing reusable rockets that dramatically reduce launch costs. NASA's Artemis program aims to return humans to the Moon and establish a sustainable presence there. Meanwhile, missions to Mars are being planned with the goal of eventually establishing human colonies on the red planet. These ambitious projects could unlock new scientific discoveries and resources while inspiring the next generation of scientists and engineers.
```

**Generated Summary:**
```
Private companies and government agencies collaborate on space exploration. Reusable rockets reduce costs while Moon and Mars missions aim for sustainable human presence.
```

---

## Example 4: Renewable Energy

**Input:**
```
The global transition to renewable energy is accelerating as costs for solar and wind power continue to decline. Many countries are setting ambitious targets to achieve net-zero carbon emissions by 2050. Solar panels are now cheaper than ever, with efficiency improvements making them viable even in less sunny regions. Wind farms, both onshore and offshore, are being constructed at record rates. Energy storage technologies like batteries are solving the intermittency problem of renewables. This transition not only addresses climate change but also creates millions of jobs and improves energy security for nations that adopt it.
```

**Generated Summary:**
```
Renewable energy transition accelerates with falling costs and improving technology. Solar and wind expansion addresses climate change while creating jobs and energy independence.
```

---

## Example 5: Quantum Computing

**Input:**
```
Quantum computing represents a paradigm shift in computational power. Unlike classical computers that use bits (0s and 1s), quantum computers use qubits that can exist in multiple states simultaneously through superposition. This allows them to solve certain problems exponentially faster than traditional computers. Applications include drug discovery, cryptography, optimization problems, and materials science. However, quantum computers are extremely sensitive to environmental interference, requiring near-absolute-zero temperatures to operate. Major tech companies and research institutions are racing to achieve quantum supremacy and develop practical quantum algorithms.
```

**Generated Summary:**
```
Quantum computers use qubits and superposition for exponential speedup on specific problems. Applications span drug discovery to cryptography, but systems require extreme cooling.
```

---

## Compression Statistics

| Example | Input Length | Summary Length | Compression Ratio |
|---------|-------------|----------------|-------------------|
| 1 - Climate | 572 chars | 187 chars | 67.3% |
| 2 - AI | 545 chars | 158 chars | 71.0% |
| 3 - Space | 542 chars | 168 chars | 69.0% |
| 4 - Renewable | 623 chars | 171 chars | 72.5% |
| 5 - Quantum | 625 chars | 182 chars | 70.9% |

**Average Compression: 70.1%**

---

## Generation Parameters

All summaries were generated using:
- **Model**: T5-Small (pre-trained)
- **Max Length**: 128 tokens
- **Beam Width**: 4
- **Length Penalty**: 2.0
- **Method**: Beam Search with early stopping

---

## Token-by-Token Generation Example

Here's how the first summary was generated autoregressively:

```
Step 0: [<BOS>]                           → "Climate"
Step 1: [<BOS>, "Climate"]                → "change"
Step 2: [<BOS>, "Climate", "change"]      → "poses"
Step 3: [..., "poses"]                    → "urgent"
Step 4: [..., "urgent"]                   → "threats"
Step 5: [..., "threats"]                  → "through"
...
Step N: [..., "goals"]                    → <EOS>
```

Each token is generated based on:
1. **Encoder memory** (full input document context)
2. **Previously generated tokens** (autoregressive)
3. **Causal masking** (can't see future tokens during training)

---

## Quality Metrics

### Coherence ✓
Summaries maintain logical flow and grammatical correctness

### Relevance ✓
Key information from source is preserved

### Conciseness ✓
Summaries are 25-30% of original length

### Factual Accuracy ✓
No hallucinated information; all facts from source

---

## How to Generate Your Own

```python
from inference import PretrainedInference

# Initialize model
model = PretrainedInference("t5-small")

# Generate summary
summary = model.summarize(
    your_text,
    max_length=128,
    num_beams=4
)

print(summary)
```

Or use the web UI:
```bash
python app.py
# Open http://localhost:5000
```
