module.exports = async (_req, res) => {
  // AI detector is optional in the local Flask backend.
  // On Vercel, we run serverless functions, so we return "unavailable" by default.
  res.status(200).json({
    available: false,
    reason: 'AI detector not deployed on Vercel serverless API',
  });
};
