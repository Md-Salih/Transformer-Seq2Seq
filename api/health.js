module.exports = async (_req, res) => {
  res.status(200).json({
    status: 'ok',
    runtime: 'vercel-serverless',
    has_hf_token: Boolean(process.env.HF_API_TOKEN),
  });
};
