const { clampInt, hfSummarize } = require('../_lib/hf');

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

module.exports = async (req, res) => {
  if (req.method !== 'POST') {
    res.status(405).json({ error: 'Method not allowed' });
    return;
  }

  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache, no-transform');
  res.setHeader('Connection', 'keep-alive');

  try {
    const { text, max_length } = req.body || {};
    const input = (text || '').toString().trim();

    if (!input) {
      res.write(`data: ${JSON.stringify({ error: 'No text provided' })}\n\n`);
      res.end();
      return;
    }

    const maxWords = clampInt(max_length, 50, 300, 150);

    // Generate once for quality, then stream as SSE tokens.
    const summary = await hfSummarize({ text: input, maxWords });

    const words = summary.split(/\s+/).filter(Boolean);
    const total = Math.max(1, words.length);

    for (let i = 0; i < words.length; i += 1) {
      const progress = Math.min(100, Math.round(((i + 1) / total) * 100));
      res.write(
        `data: ${JSON.stringify({ token: words[i] + ' ', step: i + 1, progress, done: false })}\n\n`
      );
      // Small delay for UI animation; keep low for serverless limits.
      // If you want faster, reduce to 0.
      await sleep(5);
    }

    res.write(`data: ${JSON.stringify({ done: true, progress: 100 })}\n\n`);
    res.end();
  } catch (e) {
    res.write(`data: ${JSON.stringify({ error: e?.message || 'Server error' })}\n\n`);
    res.end();
  }
};
