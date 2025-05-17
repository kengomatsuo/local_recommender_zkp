const FEED = document.getElementById('feed');
const LOADER = document.getElementById('loader');
let posts = [];
let current = 0;
let batch = [];
const BATCH_SIZE = 10;

// --- ML Model Integration (TensorFlow.js) ---
let tfModel = null;
let modelTrained = false;
let modelTraining = false;
let modelTrainingPromise = null; // Promise that resolves when training is done
let MODEL_TOPICS = [];
let MODEL_HASHTAGS = [];

// Maintain a dynamic set of topics/hashtags seen in the feed
let seenTopics = new Set();
let seenHashtags = new Set();

function updateSeenTopicsAndHashtags(postsBatch) {
  for (const post of postsBatch) {
    (post.topics || []).forEach(t => seenTopics.add(t));
    (post.hashtags || []).forEach(h => seenHashtags.add(h));
  }
  MODEL_TOPICS = Array.from(seenTopics);
  MODEL_HASHTAGS = Array.from(seenHashtags);
}

async function loadModel(force = false) {
  // Wait for training to finish before reinitializing
  if (modelTraining && modelTrainingPromise) {
    await modelTrainingPromise;
  }
  if (!tfModel || force) {
    if (MODEL_TOPICS.length === 0) {
      tfModel = null;
      modelTrained = false;
      return;
    }
    tfModel = tf.sequential();
    tfModel.add(tf.layers.dense({inputShape: [MODEL_TOPICS.length], units: 3, activation: 'softmax'}));
    tfModel.compile({optimizer: 'adam', loss: 'categoricalCrossentropy'});
    modelTrained = false;
  }
}

async function trainModel(interactionsObj) {
  const interactions = Object.values(interactionsObj);
  if (MODEL_TOPICS.length === 0) return;
  await loadModel(true); // Always refresh topics/hashtags before training
  if (interactions.length < 5) return;
  modelTraining = true;
  let resolveTraining;
  modelTrainingPromise = new Promise(r => resolveTraining = r);
  await renderModelStatus(); // show training status
  try {
    // Only use the last 50 interactions for training
    const recentInteractions = interactions.slice(-100);
    const xs = [];
    const ys = [];
    for (const inter of recentInteractions) {
      // Engagement classification logic
      let engaged = 1; // default: partially engaged
      const interest_flag = inter.interested ? 'Interested' : (inter.not_interested ? 'Not Interested' : null);
      const liked = !!inter.liked;
      const commented = !!inter.commented;
      const time_watched = inter.timeSpentMs || 0;
      const duration = inter.duration || 10000; // fallback duration (ms) if not present
      if (interest_flag === 'Interested') {
        engaged = 2;
      } else if (
        interest_flag === 'Not Interested' ||
        ((time_watched / duration) < 0.25 && !liked && !commented)
      ) {
        engaged = 0;
      } else {
        engaged = ((time_watched / duration) > 0.8 || liked || commented) ? 2 : 1;
      }
      // For ML: one-hot encode engagement (0, 1, 2)
      const x = MODEL_TOPICS.map(t => (inter.topics || []).includes(t) ? 1 : 0);
      const y = [0, 0, 0];
      y[engaged] = 1;
      xs.push(x);
      ys.push(y);
    }
    if (xs.length === 0 || xs[0].length === 0) return;
    const xsTensor = tf.tensor2d(xs);
    const ysTensor = tf.tensor2d(ys);
    await tfModel.fit(xsTensor, ysTensor, {epochs: 10, batchSize: 4});
    xsTensor.dispose();
    ysTensor.dispose();
    modelTrained = true;
    renderInteractionHistory();
  } finally {
    modelTraining = false; // NEW: clear training flag
    if (resolveTraining) resolveTraining();
    modelTrainingPromise = null;
    await renderModelStatus(); // show trained/untrained status
  }
}

async function analyzeInteractions(interactionsObj) {
  const interactions = Object.values(interactionsObj);
  if (MODEL_TOPICS.length === 0) {
    return { topics: [], hashtags: [] };
  }
  await loadModel(true); // Always refresh topics/hashtags before prediction
  // if (!modelTrained || !tfModel) {
  //   return { topics: [], hashtags: [] };
  // }
  const topicCounts = MODEL_TOPICS.map(t =>
    interactions.reduce((sum, i) => sum + ((i.topics || []).includes(t) ? 1 : 0), 0)
  );
  const input = tf.tensor([topicCounts]);
  const pred = tfModel.predict(input).arraySync();
  input.dispose();
  // pred is [ [engaged0, engaged1, engaged2], ... ]
  // For each topic, show the most likely engagement class (argmax)
  const scoredTopics = MODEL_TOPICS.map((t, i) => {
    // For each topic, get the engagement class with highest probability
    // Here, we use the engagement=2 (high) probability for ranking
    return { t, score: pred[0][2] };
  });
  scoredTopics.sort((a, b) => b.score - a.score);
  // Plot topic scores for debugging
  if (typeof window !== 'undefined') {
    const plotDiv = document.getElementById('topic-plot') || document.createElement('div');
    plotDiv.id = 'topic-plot';
    plotDiv.style = 'position:fixed;bottom:0;left:340px;right:340px;background:#fff;border-top:1px solid #ccc;padding:10px;z-index:1001;max-width:calc(100vw - 680px);overflow-x:auto;';
    let html = `<b>Topic Scores</b><br><div style='display:flex;align-items:flex-end;height:80px;'>`;
    for (const s of scoredTopics) {
      html += `<div style='margin:0 4px;text-align:center;'><div style='background:#4af;width:18px;height:${Math.round(s.score*60)}px;'></div><div style='font-size:0.8em;max-width:40px;overflow:hidden;text-overflow:ellipsis;'>${s.t}</div><div style='font-size:0.8em;'>${s.score.toFixed(2)}</div></div>`;
    }
    html += '</div>';
    plotDiv.innerHTML = html;
    document.body.appendChild(plotDiv);
  }
  let splitIdx = scoredTopics.length;
  let maxGap = 0;
  for (let i = 0; i < scoredTopics.length - 1; i++) {
    const gap = scoredTopics[i].score - scoredTopics[i + 1].score;
    if (gap > maxGap && scoredTopics[i].score > 0.1) {
      maxGap = gap;
      splitIdx = i + 1;
    }
  }
  const topTopics = scoredTopics.slice(0, splitIdx).filter(e => e.score > 0.1).map(e => e.t);

  // For hashtags, use frequency and group by largest drop
  const hashtagCounts = {};
  for (const inter of interactions) {
    if (inter.hashtags) inter.hashtags.forEach(h => hashtagCounts[h] = (hashtagCounts[h] || 0) + 1);
  }
  const scoredHashtags = Object.entries(hashtagCounts).map(([h, count]) => ({ h, count }));
  scoredHashtags.sort((a, b) => b.count - a.count);
  let splitHIdx = scoredHashtags.length;
  let maxHGap = 0;
  for (let i = 0; i < scoredHashtags.length - 1; i++) {
    const gap = scoredHashtags[i].count - scoredHashtags[i + 1].count;
    if (gap > maxHGap && scoredHashtags[i].count > 0) {
      maxHGap = gap;
      splitHIdx = i + 1;
    }
  }
  const topHashtags = scoredHashtags.slice(0, splitHIdx).filter(e => e.count > 0).map(e => e.h);
  return { topics: topTopics, hashtags: topHashtags };
}

// Change interactions to a map for per-post interaction
let interactions = {};
let idleTimeout = null;
const IDLE_TIME_MS = 10000; // 10 seconds

function resetIdleTimer() {
  if (idleTimeout) clearTimeout(idleTimeout);
  idleTimeout = setTimeout(async () => {
    await trainModel(interactions);
  }, IDLE_TIME_MS);
}

// Patch recordInteraction to update per-post interaction
recordInteraction = async function(type, post) {
  // Ensure interaction object exists for this post
  if (!post.id) return;
  if (!interactions[post.id]) {
    interactions[post.id] = {
      postId: post.id,
      topics: post.topics,
      hashtags: post.hashtags,
      liked: false,
      interested: false,
      not_interested: false,
      commented: false
    };
  }
  // Update interaction state
  if (type === 'interested') {
    interactions[post.id].interested = true;
    interactions[post.id].not_interested = false;
  } else if (type === 'not_interested') {
    interactions[post.id].not_interested = true;
    interactions[post.id].interested = false;
  }
  interactions[post.id].timestamp = Date.now();
  // If this is the last post in the batch, train the model
  if (batch.length && batch[current] && post.id === batch[batch.length - 1].id && current === batch.length - 1) {
    trainModel(interactions);
  }
  resetIdleTimer();
};

async function loadBatch() {
  // Analyze interactions before requesting new batch
  const { topics, hashtags } = await analyzeInteractions(interactions);
  const params = new URLSearchParams();
  if (topics.length) params.append('topics', topics.join(','));
  if (hashtags.length) params.append('hashtags', hashtags.join(','));
  params.append('limit', BATCH_SIZE);
  const res = await fetch('http://localhost:3000/api/posts?' + params.toString());
  const data = await res.json();
  batch = data.posts;
  updateSeenTopicsAndHashtags(batch);
  console.log('Loaded batch:', batch);
  current = 0;
  // Removed model training from here
}

function createPost(post) {
  const div = document.createElement('div');
  div.className = 'post';
  const content = document.createElement('div');
  content.className = 'content';
  content.innerHTML = `
    <strong>${post.title || ''}</strong>
    <p>${post.body || ''}</p>
    <div><b>Topics:</b> ${(post.topics && post.topics.length) ? post.topics.join(', ') : 'None'}</div>
    <div><b>Hashtags:</b> ${(post.hashtags && post.hashtags.length) ? post.hashtags.join(', ') : 'None'}</div>
    <div style="color:#888;font-size:0.9em;margin-top:8px;">${post.id || ''}</div>
  `;
  div.appendChild(content);
  div.appendChild(addInteractionButtons(div, post)); // Fix: actually append buttons
  return div;
}

// Add UI for like, interested, not interested
function addInteractionButtons(div, post) {
  const btns = document.createElement('div');
  btns.className = 'interaction-btns';
  // Get current interaction state for this post
  const inter = interactions[post.id] || {};
  // Like button: toggle and show state
  const likeActive = inter.liked ? 'background:#cfc;' : '';
  // Comment button: disabled if already commented
  const commentDisabled = inter.commented ? 'disabled' : '';
  btns.innerHTML = `
    <button id="like-btn" style="${likeActive}">👍 Like</button>
    <button id="interested-btn">Interested</button>
    <button id="not-interested-btn">Not Interested</button>
    <button id="comment-btn" ${commentDisabled}>💬 Comment</button>
  `;
  btns.querySelector('#like-btn').onclick = async () => {
    // Toggle like
    if (!interactions[post.id]) interactions[post.id] = { postId: post.id, topics: post.topics, hashtags: post.hashtags, liked: false, interested: false, not_interested: false, commented: false, timestamp: Date.now() };
    interactions[post.id].liked = !interactions[post.id].liked;
    interactions[post.id].timestamp = Date.now();
    // Re-render buttons to update like state
    const newBtns = addInteractionButtons(div, post);
    btns.replaceWith(newBtns);
  };
  btns.querySelector('#interested-btn').onclick = () => recordInteraction('interested', post);
  btns.querySelector('#not-interested-btn').onclick = () => recordInteraction('not_interested', post);
  btns.querySelector('#comment-btn').onclick = async () => {
    if (!interactions[post.id]) interactions[post.id] = { postId: post.id, topics: post.topics, hashtags: post.hashtags, liked: false, interested: false, not_interested: false, commented: false, timestamp: Date.now() };
    if (!interactions[post.id].commented) {
      interactions[post.id].commented = true;
      interactions[post.id].timestamp = Date.now();
      // Re-render buttons to disable comment
      const newBtns = addInteractionButtons(div, post);
      btns.replaceWith(newBtns);
    }
  };
  return btns;
}

let postViewStartTime = null;
let lastViewedPostId = null;

function recordTimeSpentOnCurrentPost() {
  if (lastViewedPostId && postViewStartTime) {
    const now = Date.now();
    const spent = now - postViewStartTime;
    if (!interactions[lastViewedPostId]) {
      // If interaction doesn't exist, create a new one with minimal info
      interactions[lastViewedPostId] = {
        postId: lastViewedPostId,
        timeSpentMs: spent,
        timestamp: now
      };
    } else {
      if (!interactions[lastViewedPostId].timeSpentMs) {
        interactions[lastViewedPostId].timeSpentMs = 0;
      }
      interactions[lastViewedPostId].timeSpentMs += spent;
    }
  }
}

async function showPost(idx) {
  recordTimeSpentOnCurrentPost();
  if (batch.length === 0 || idx < 0 || idx >= batch.length - 2) {
    await loadBatch();
    idx = 0;
  }
  FEED.innerHTML = '';
  FEED.appendChild(createPost(batch[idx]));
  postViewStartTime = Date.now();
  lastViewedPostId = batch[idx].id;
  resetIdleTimer();
  renderInteractionHistory();
  await renderModelStatus();
}

function nextPost() {
  recordTimeSpentOnCurrentPost();
  if (current < batch.length - 1) {
    current++;
    showPost(current);
  } else {
    loadBatch().then(() => showPost(0));
  }
}
function prevPost() {
  recordTimeSpentOnCurrentPost();
  if (current > 0) {
    current--;
    showPost(current);
  }
}

function onKey(e) {
  console.log(e.key);
  if (e.key === 'ArrowDown' || e.key === ' ') nextPost();
  else if (e.key === 'ArrowUp') prevPost();
}

window.addEventListener('wheel', null, { passive: true });
window.addEventListener('keydown', onKey);
window.addEventListener('DOMContentLoaded', () => showPost(current));

// --- UI for interaction history and topics ---
const INTERACTIONS_DIV = document.createElement('div');
INTERACTIONS_DIV.id = 'interactions-history';
INTERACTIONS_DIV.style = 'position:fixed;right:0;top:0;width:340px;max-height:100vh;overflow:auto;background:#fff;border-left:1px solid #ccc;padding:10px;font-size:0.95em;z-index:1000;box-shadow:-2px 0 8px #0001;';
document.body.appendChild(INTERACTIONS_DIV);

// Update renderInteractionHistory to reflect new structure
function renderInteractionHistory() {
  let html = '<b>User Interactions</b><br><ul style="margin:0 0 10px 0;padding-left:18px;max-height:180px;overflow:auto;">';
  const allInteractions = Object.values(interactions).sort((a, b) => b.timestamp - a.timestamp);
  for (const inter of allInteractions) {
    let details = [];
    if (inter.liked) details.push('👍');
    if (inter.interested) details.push('Interested');
    if (inter.not_interested) details.push('Not Interested');
    if (inter.commented) details.push('💬');
    let timeStr = `<div style='color:#0a7;font-size:0.95em;margin:2px 0 2px 0;'><b>Time spent:</b> ${Math.round(inter.timeSpentMs/1000)}s</div>`;
    html += `<li><b>${details.join(', ') || 'None'}</b> <span style='color:#888'>[${inter.postId}]</span>${timeStr}<br><span style='color:#555'>${(inter.topics||[]).join(', ')}</span> <span style='color:#888'>${(inter.hashtags||[]).join(', ')}</span></li>`;
  }
  html += '</ul>';
  html += '<b>Current Topics Array</b><br>';
  html += `<div style='word-break:break-all;color:#1a4'>[${MODEL_TOPICS.map(t => `'${t}'`).join(', ')}]</div>`;
  INTERACTIONS_DIV.innerHTML = html;
}

// Patch recordInteraction to update UI
const _recordInteraction = recordInteraction;
recordInteraction = function(type, post) {
  _recordInteraction(type, post);
};

// Also update UI on page load and after topics update
window.addEventListener('DOMContentLoaded', () => {
  renderInteractionHistory();
});

// --- UI for model status and next API params ---
const MODEL_STATUS_DIV = document.createElement('div');
MODEL_STATUS_DIV.id = 'model-status';
MODEL_STATUS_DIV.style = 'position:fixed;left:0;top:0;width:340px;max-width:100vw;max-height:100vh;overflow:auto;background:#fff;border-right:1px solid #ccc;padding:10px;font-size:0.95em;z-index:1000;box-shadow:2px 0 8px #0001;';
document.body.appendChild(MODEL_STATUS_DIV);

let lastAnalyzed = { topics: [], hashtags: [] };

async function renderModelStatus() {
  let status = modelTraining
    ? '<span style="color:orange">Training</span>'
    : (modelTrained ? '<span style="color:green">Trained</span>' : '<span style="color:red">Untrained</span>');
  let html = `<b>Model Status:</b> ${status}<br>`;
  html += `<b>Topics Array</b><br><div style='word-break:break-all;color:#1a4'>[${MODEL_TOPICS.map(t => `'${t}'`).join(', ')}]</div>`;
  if (modelTraining) {
    html += `<b>Next API Topics</b><br><div style='color:#aaa;font-style:italic;'>Updating...</div>`;
    html += `<b>Next API Hashtags</b><br><div style='color:#aaa;font-style:italic;'>Updating...</div>`;
  } else {
    lastAnalyzed = await analyzeInteractions(interactions);
    html += `<b>Next API Topics</b><br><div style='word-break:break-all;color:#14a'>[${(lastAnalyzed.topics||[]).map(t => `'${t}'`).join(', ')}]</div>`;
    html += `<b>Next API Hashtags</b><br><div style='word-break:break-all;color:#14a'>[${(lastAnalyzed.hashtags||[]).map(h => `'${h}'`).join(', ')}]</div>`;
  }
  MODEL_STATUS_DIV.innerHTML = html;
}

// Patch loadBatch to train model before loading new batch and then update model status
const _loadBatch = loadBatch;
loadBatch = async function() {
  await trainModel(interactions); // Train BEFORE loading new batch
  await _loadBatch();
  await renderModelStatus();
};

// Also update UI on page load and after topics update
window.addEventListener('DOMContentLoaded', async () => {
  renderInteractionHistory();
  await renderModelStatus();
});

// Patch recordInteraction to update UI and model status (merge with existing patch logic)
recordInteraction = async function(type, post) {
  // Ensure interaction object exists for this post
  if (!post.id) return;
  if (!interactions[post.id]) {
    interactions[post.id] = {
      postId: post.id,
      topics: post.topics,
      hashtags: post.hashtags,
      liked: false,
      interested: false,
      not_interested: false,
      commented: false,
      timestamp: Date.now()
    };
  }
  if (type === 'interested') {
    interactions[post.id].interested = true;
    interactions[post.id].not_interested = false;
  } else if (type === 'not_interested') {
    interactions[post.id].not_interested = true;
    interactions[post.id].interested = false;
  }
  if (lastViewedPostId === post.id && postViewStartTime) {
    const now = Date.now();
    const spent = now - postViewStartTime;
    if (!interactions[post.id].timeSpentMs) {
      interactions[post.id].timeSpentMs = 0;
    }
    interactions[post.id].timeSpentMs += spent;
    postViewStartTime = now;
  }
  interactions[post.id].timestamp = Date.now();
  // No UI update here
  if (batch.length && batch[current] && post.id === batch[batch.length - 1].id && current === batch.length - 1) {
    trainModel(interactions);
  }
  resetIdleTimer();
};
