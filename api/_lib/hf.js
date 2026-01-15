const DEFAULT_MODEL = process.env.HF_MODEL || 'google/flan-t5-small';

function requireEnv(name) {
  const value = process.env[name];
  if (!value) {
    const error = new Error(`Missing required environment variable: ${name}`);
    error.statusCode = 500;
    throw error;
  }
  return value;
}

function clampInt(value, min, max, fallback) {
  const n = Number.parseInt(value, 10);
  if (Number.isNaN(n)) return fallback;
  return Math.max(min, Math.min(max, n));
}

// Rough conversion: words -> tokens for T5-family.
// Kept conservative to stay within serverless time limits.
function wordsToMaxNewTokens(maxWords) {
  // 1 word ~ 1.3 tokens on average; keep a floor/ceiling.
  return Math.max(32, Math.min(256, Math.round(maxWords * 1.3)));
}

async function hfSummarize({ text, maxWords }) {
  const token = requireEnv('HF_API_TOKEN');

  const maxNewTokens = wordsToMaxNewTokens(maxWords);
  const minNewTokens = Math.max(16, Math.min(192, Math.round(maxNewTokens * 0.45)));

  const prompt = `summarize: ${text}`;

  const res = await fetch(`https://api-inference.huggingface.co/models/${encodeURIComponent(DEFAULT_MODEL)}`, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${token}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      inputs: prompt,
      parameters: {
        max_new_tokens: maxNewTokens,
        min_new_tokens: minNewTokens,
        do_sample: false,
        num_beams: 4,
        no_repeat_ngram_size: 3,
        repetition_penalty: 1.15,
      },
      options: {
        wait_for_model: true,
      },
    }),
  });

  const contentType = res.headers.get('content-type') || '';
  const raw = contentType.includes('application/json') ? await res.json() : await res.text();

  if (!res.ok) {
    const msg = typeof raw === 'string' ? raw : JSON.stringify(raw);
    const error = new Error(`HuggingFace inference error (${res.status}): ${msg}`);
    error.statusCode = res.status;
    throw error;
  }

  // HF returns array like: [{ generated_text: '...' }]
  const generated = Array.isArray(raw) ? raw?.[0]?.generated_text : raw?.generated_text;
  if (typeof generated !== 'string' || !generated.trim()) {
    const error = new Error('HuggingFace returned no generated_text.');
    error.statusCode = 502;
    throw error;
  }

  return generated.trim();
}

module.exports = {
  clampInt,
  hfSummarize,
};
