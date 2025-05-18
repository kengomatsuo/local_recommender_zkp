import express from 'express';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = 3000;

// Load posts once at startup
const POSTS_PATH = path.join(__dirname, 'mock_posts_200.json');
let posts = [];
try {
  posts = JSON.parse(fs.readFileSync(POSTS_PATH, 'utf8'));
} catch (e) {
  console.error('Failed to load posts:', e);
}

// Enable CORS for local dev
app.use((req, res, next) => {
  const allowedOrigins = ['http://localhost:3000', 'http://127.0.0.1:3000', 'http://localhost:5173', 'http://localhost:4000'];
  const origin = req.headers.origin;

  if (allowedOrigins.includes(origin)) {
    res.header('Access-Control-Allow-Origin', origin);
  }

  res.header('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept, X-ZKP-Proof, X-ZKP-PublicSignals');
  res.header('Access-Control-Allow-Methods', 'GET,POST,OPTIONS');

  if (req.method === 'OPTIONS') {
    return res.sendStatus(200);
  }

  next();
});


app.use(express.json()); // Add this to parse JSON bodies

// Serve static files (for index.html, main.js, etc.)
app.use(express.static(__dirname));

// API endpoint for random batch of posts
app.get('/api/posts', (req, res) => {
  res.header('Access-Control-Allow-Origin', '*'); // Allow all origins for API
  const limit = parseInt(req.query.limit) || 10;
  let filtered = posts;

  // Accept weights as JSON in query or POST body
  let userTopics = [];
  let userHashtags = [];
  try {
    if (req.query.topics) {
      userTopics = JSON.parse(req.query.topics);
    } else if (req.body && req.body.topics) {
      userTopics = req.body.topics;
    }
  } catch (e) {
    // fallback to comma-separated string
    userTopics = req.query.topics ? req.query.topics.split(',').map(name => ({ name, weight: 1 })) : [];
  }
  try {
    if (req.query.hashtags) {
      userHashtags = JSON.parse(req.query.hashtags);
    } else if (req.body && req.body.hashtags) {
      userHashtags = req.body.hashtags;
    }
  } catch (e) {
    userHashtags = req.query.hashtags ? req.query.hashtags.split(',').map(name => ({ name, weight: 1 })) : [];
  }

  // If weights provided, score posts
  if ((userTopics.length && typeof userTopics[0] === 'object') || (userHashtags.length && typeof userHashtags[0] === 'object')) {
    // Build lookup maps for fast scoring
    const topicWeightMap = Object.fromEntries(userTopics.map(t => [t.name, t.weight]));
    const hashtagWeightMap = Object.fromEntries(userHashtags.map(h => [h.name, h.weight]));
    // Score each post
    filtered = posts.map(post => {
      let score = 0;
      if (post.topics && post.topics.length) {
        score += post.topics.reduce((sum, t) => sum + (topicWeightMap[t] || 0), 0);
      }
      if (post.hashtags && post.hashtags.length) {
        score += post.hashtags.reduce((sum, h) => sum + (hashtagWeightMap[h] || 0), 0);
      }
      return { post, score };
    })
    .sort((a, b) => b.score - a.score)
    .slice(0, limit * 2) // take more for noise
    .map(obj => obj.post);
    // Add noise: randomly add some unrelated posts
    const noiseCount = Math.floor(limit * 0.3);
    const noise = posts.filter(post => !filtered.includes(post))
      .sort(() => 0.5 - Math.random())
      .slice(0, noiseCount);
    filtered = filtered.concat(noise);
  } else if (userTopics.length || userHashtags.length) {
    // Old fallback: match by name only
    filtered = posts.filter(post => {
      let topicMatch = userTopics.length && post.topics && post.topics.some(t => userTopics.includes(t.name || t));
      let hashtagMatch = userHashtags.length && post.hashtags && post.hashtags.some(h => userHashtags.includes(h.name || h));
      return topicMatch || hashtagMatch;
    });
    // Add noise
    const noiseCount = Math.floor(limit * 0.3);
    const noise = posts.filter(post => !filtered.includes(post))
      .sort(() => 0.5 - Math.random())
      .slice(0, noiseCount);
    filtered = filtered.concat(noise);
  }

  // Shuffle and return a random batch
  const shuffled = filtered.slice().sort(() => 0.5 - Math.random());
  const batch = shuffled.slice(0, limit);
  res.json({
    posts: batch,
    limit,
    total: posts.length
  });
});

// Load verification key
const vKey = JSON.parse(fs.readFileSync(path.join(__dirname, "keys", "vk.json")));

// Verify ZKP before allowing post fetch
app.use('/api/posts', async (req, res, next) => {
  const proofHeader = req.headers['x-zkp-proof'];
  const publicSignalsHeader = req.headers['x-zkp-publicsignals'];

  if (!proofHeader || !publicSignalsHeader) {
    console.warn("Missing ZKP headers");
    return res.status(401).json({ error: 'Authentication required' });
  }

  try {
    const proof = JSON.parse(proofHeader);
    const publicSignals = JSON.parse(publicSignalsHeader);

    const isValid = await snarkjs.groth16.verify(vKey, publicSignals, proof);

    if (!isValid) {
      console.warn("Invalid ZKP");
      return res.status(403).json({ error: 'Invalid authentication proof' });
    }

    // Check Merkle root matches expected
    const expectedRoot = "3659df8a4bfb87fbf444433c10d49d0e6434ac6c18072d9eaf14802efe00a5e5"; // Update if your root changes
    if (publicSignals[0] !== expectedRoot) {
      console.warn("Merkle root mismatch");
      return res.status(403).json({ error: 'Merkle root mismatch' });
    }
    console.log("ZKP verified successfully");
    // Attach verified flag
    req.authenticated = true;
    next();
  } catch (error) {
    console.error("Error verifying ZKP:", error);
    return res.status(400).json({ error: 'Malformed authentication data' });
  }
});

app.listen(PORT, () => {
  console.log(`Server running at http://localhost:${PORT}`);
});
