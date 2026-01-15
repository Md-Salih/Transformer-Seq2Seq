const { clampInt, hfSummarize } = require('../_lib/hf');

module.exports = async (req, res) => {
  if (req.method !== 'POST') {
    res.status(405).json({ error: 'Method not allowed' });
    return;
  }

  try {
    const { text, max_length } = req.body || {};

    const input = (text || '').toString().trim();
    if (!input) {
      res.status(400).json({ error: 'No text provided' });
      return;
    }

    // Frontend slider is "words"; keep same param name for compatibility.
    const maxWords = clampInt(max_length, 50, 300, 150);

    const summary = await hfSummarize({ text: input, maxWords });

    res.status(200).json({
      summary,
      max_length: maxWords,
      engine: 'hf-inference-api',
    });
  } catch (e) {
    const status = e?.statusCode && Number.isInteger(e.statusCode) ? e.statusCode : 500;
    res.status(status).json({ error: e?.message || 'Server error' });
  }
};
