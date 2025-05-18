import { previousTopicsResults, previousHashtagsResults } from "./model.js";

// DOM Elements
const FEED = document.getElementById("feed");
const LOADER = document.getElementById("loader");

// Feed State
let posts = [];
let current = 0;
let batch = [];
const BATCH_SIZE = 10;

// ML Model State
import {
  tfModel,
  modelTrained,
  modelTraining,
  modelTrainingPromise,
  MODEL_TOPICS,
  MODEL_HASHTAGS,
  loadModel,
  trainModel,
  analyzeInteractions,
  analyzeWithoutModel,
  findNaturalSplit,
  WEIGHT_LIKED,
  WEIGHT_INTERESTED,
  WEIGHT_NOT_INTERESTED,
  WEIGHT_COMMENTED,
  MIN_INTERACTIONS,
} from "./model.js";
import { fetchPosts, generateZKPProof } from "./api.js";

// Preference weighting constants

// User Interaction State
let interactions = [];
let postViewStartTime = null;
let lastViewedPostId = null;

// UI Elements
const INTERACTIONS_DIV = document.createElement("div");
const MODEL_STATUS_DIV = document.createElement("div");
let lastAnalyzed = { topics: [], hashtags: [] };

// Debounce/throttle configuration
let idleTimeout = null;
const IDLE_TIME_MS = 10000; // 10 seconds
let uiUpdateDebounceTimer = null;
const UI_DEBOUNCE_TIME = 250; // 250ms debounce for UI updates

// Initialize UI and listeners
function initializeApp() {
  setupUIElements();
  setupEventListeners();
  showPost(current);
}

// Initialize UI elements
function setupUIElements() {
  // Interaction history sidebar
  INTERACTIONS_DIV.id = "interactions-history";
  INTERACTIONS_DIV.style =
    "position:fixed;right:0;top:0;width:340px;height:100vh;overflow:auto;background:#fff;border-left:1px solid #ccc;padding:10px;font-size:0.95em;z-index:1;box-shadow:-2px 0 8px #0001;";
  document.body.appendChild(INTERACTIONS_DIV);

  // Model status sidebar
  MODEL_STATUS_DIV.id = "model-status";
  MODEL_STATUS_DIV.style =
    "position:fixed;left:0;top:0;width:340px;max-width:100vw;height:100vh;overflow:auto;background:#fff;border-right:1px solid #ccc;padding:10px;font-size:0.95em;z-index:1;box-shadow:2px 0 8px #0001;";
  document.body.appendChild(MODEL_STATUS_DIV);
}

// Event listeners
function setupEventListeners() {
  window.addEventListener("keydown", onKey);

  // Use passive listeners for better scrolling performance
  window.addEventListener("wheel", () => {}, { passive: true });

  // Handle application visibility changes
  document.addEventListener("visibilitychange", handleVisibilityChange);
}

// Centralized topic and hashtag tracking
function updateSeenTopicsAndHashtags(postsBatch) {
  // Use Sets to track unique topics/hashtags
  const seenTopics = new Set(MODEL_TOPICS);
  const seenHashtags = new Set(MODEL_HASHTAGS);

  // Update with new posts
  for (const post of postsBatch) {
    (post.topics || []).forEach((t) => seenTopics.add(t));
    (post.hashtags || []).forEach((h) => seenHashtags.add(h));
  }

  // Mutate arrays in place instead of reassigning (fixes Assignment to constant variable)
  MODEL_TOPICS.splice(0, MODEL_TOPICS.length, ...Array.from(seenTopics));
  MODEL_HASHTAGS.splice(0, MODEL_HASHTAGS.length, ...Array.from(seenHashtags));
}

// Record interactions with improved logging
async function recordInteraction(type, post) {
  if (!post || !post.id) return;

  // Find interaction object for this post
  let inter = interactions.find((i) => i.postId === post.id);
  if (!inter) {
    inter = {
      postId: post.id,
      topics: post.topics || [],
      hashtags: post.hashtags || [],
      liked: false,
      interested: false,
      not_interested: false,
      commented: false,
      timeSpentMs: 0,
      timestamp: Date.now(),
    };
    interactions.push(inter);

    // Trim to keep only the last 100 interactions
    if (interactions.length > 100) {
      interactions = interactions.slice(-100);
    }
  }

  // Previous state for logging
  const prevState = {
    liked: inter.liked,
    interested: inter.interested,
    not_interested: inter.not_interested,
  };

  // Update interaction state based on type
  if (type === "interested") {
    inter.interested = true;
    inter.not_interested = false; // Make these mutually exclusive
  } else if (type === "not_interested") {
    inter.not_interested = true;
    inter.interested = false; // Make these mutually exclusive
  } else if (type === "like") {
    inter.liked = !inter.liked;
  } else if (type === "comment") {
    inter.commented = true;
  }

  // Log interaction change for debugging
  console.log(`Interaction updated for post ${post.id}:`, {
    type,
    prevState,
    newState: {
      liked: inter.liked,
      interested: inter.interested,
      not_interested: inter.not_interested,
    },
  });

  // Update time spent if this is the current post
  if (lastViewedPostId === post.id && postViewStartTime) {
    recordTimeSpentOnCurrentPost();
    // Reset timer for ongoing viewing
    postViewStartTime = Date.now();
  }

  // Always update timestamp
  inter.timestamp = Date.now();

  // Schedule idle training
  resetIdleTimer();

  // Update UI
  debouncedRenderUI();
}

// Record time spent on current post
function recordTimeSpentOnCurrentPost() {
  if (!lastViewedPostId || !postViewStartTime) return;

  try {
    const now = Date.now();
    const spent = now - postViewStartTime;

    // Only count reasonable time spans (1ms to 5min)
    if (spent > 0 && spent < 300000) {
      // Find interaction object for this post
      let inter = interactions.find((i) => i.postId === lastViewedPostId);
      if (!inter) {
        inter = {
          postId: lastViewedPostId,
          topics: batch[current]?.topics || [],
          hashtags: batch[current]?.hashtags || [],
          liked: false,
          interested: false,
          not_interested: false,
          commented: false,
          timeSpentMs: 0,
          timestamp: now,
        };
        interactions.push(inter);
      }
      // Add time
      inter.timeSpentMs += spent;
    }
  } catch (error) {
    console.warn("Error recording time spent:", error);
  }
}

// Reset idle timer for model training
function resetIdleTimer() {
  if (idleTimeout) {
    clearTimeout(idleTimeout);
  }

  idleTimeout = setTimeout(async () => {
    console.log("Idle timeout reached, training model...");
    if (!modelTraining) {
      await trainModel(interactions);
    }
  }, IDLE_TIME_MS);
}

let isLoadingBatch = false;

// Load a new batch of posts
async function loadBatch() {
  if (isLoadingBatch) return;
  try {
    isLoadingBatch = true;
    // Record time spent on current post before loading new batch
    recordTimeSpentOnCurrentPost();

    // If training is not in progress, train the model
    if (!modelTraining) {
      console.log("Training model with interactions...");
      await trainModel(interactions);
    }

    // Analyze interactions for recommendations
    lastAnalyzed = await analyzeInteractions(interactions);

    // Build API parameters
    const params = {};
    if (lastAnalyzed.topics.length)
      params.topics = lastAnalyzed.topics.map((t) => t.name || t);
    if (lastAnalyzed.hashtags.length)
      params.hashtags = lastAnalyzed.hashtags.map((h) => h.name || h);
    params.limit = BATCH_SIZE;

    // Fetch posts with error handling using the API module
    try {
      const auth = await generateZKPProof(
        "ab2f20d2e957149afeda6cfa09efedcf0986809fcd6bfd4cd83fca58033f311f", // Example Leaf
        "4b197fc392cd047e0a1b4467778294600329e08f7bcd41d13cdf46779a2c494f", // Merkle Root
        [
          "e3c58b672d729be3cf3f6dca66bc9bedf0bedcae1fdb356d6c3760f5e65d4fb5",
          "ca0637690eb8669c2e89d6655ae5825a4f6b2c645be5f582f8e57166674519c3",
          "0b68be77c15ed75d63846072b00497c8d5351f884590e9a98bc65220ed53823b",
        ], // Hex Path
        [1, 1, 1] // Directions
      );

      console.log("ZKP proof generated successfully:", auth);
      const data = await fetchPosts(params, auth);
      // Update batch state
      batch = data.posts || [];
      current = 0;
      // Update topics and hashtags based on new posts
      updateSeenTopicsAndHashtags(batch);
      console.log("Loaded batch:", batch);
    } catch (fetchError) {
      console.error("Error fetching posts:", fetchError);
      // Use empty batch as fallback
      batch = [];
      current = 0;
    }
  } catch (error) {
    console.error("Error in loadBatch:", error);
  } finally {
    isLoadingBatch = false;
  }
}

// Create post DOM element
function createPost(post) {
  if (!post) return document.createElement("div");

  const div = document.createElement("div");
  div.className = "post";

  const content = document.createElement("div");
  content.className = "content";
  content.innerHTML = `
    <strong>${post.title || ""}</strong>
    <p>${post.body || ""}</p>
    <div><b>Topics:</b> ${
      post.topics && post.topics.length ? post.topics.join(", ") : "None"
    }</div>
    <div><b>Hashtags:</b> ${
      post.hashtags && post.hashtags.length ? post.hashtags.join(", ") : "None"
    }</div>
    <div style="color:#888;font-size:0.9em;margin-top:8px;">${
      post.id || ""
    }</div>
  `;
  div.appendChild(content);

  // Add interaction buttons
  const btns = createInteractionButtons(post);
  div.appendChild(btns);

  return div;
}

// Create interaction buttons for a post
function createInteractionButtons(post) {
  if (!post || !post.id) return document.createElement("div");

  const btns = document.createElement("div");
  btns.className = "interaction-btns";

  // Get current interaction state for this post
  const inter = interactions.find((i) => i.postId === post.id) || {};

  // Set button states based on interactions
  const likeActive = inter.liked ? "background:#cfc;" : "";
  const commentDisabled = inter.commented ? "disabled" : "";

  btns.innerHTML = `
    <button id="like-btn-${post.id}" style="${likeActive}">👍 Like</button>
    <button id="interested-btn-${post.id}">Interested</button>
    <button id="not-interested-btn-${post.id}">Not Interested</button>
    <button id="comment-btn-${post.id}" ${commentDisabled}>💬 Comment</button>
  `;

  // Add event listeners
  btns.querySelector(`#like-btn-${post.id}`).addEventListener("click", () => {
    recordInteraction("like", post);
    // Update button appearance immediately
    btns.querySelector(`#like-btn-${post.id}`).style.background = !inter.liked
      ? "#cfc"
      : "";
  });

  btns
    .querySelector(`#interested-btn-${post.id}`)
    .addEventListener("click", () => {
      recordInteraction("interested", post);
    });

  btns
    .querySelector(`#not-interested-btn-${post.id}`)
    .addEventListener("click", () => {
      recordInteraction("not_interested", post);
    });

  btns
    .querySelector(`#comment-btn-${post.id}`)
    .addEventListener("click", () => {
      if (!inter.commented) {
        recordInteraction("comment", post);
        // Update button state immediately
        btns.querySelector(`#comment-btn-${post.id}`).disabled = true;
      }
    });

  return btns;
}

// Show a post by index
async function showPost(idx) {
  try {
    // Record time spent on current post
    recordTimeSpentOnCurrentPost();

    // Load new batch if needed
    if (batch.length === 0 || idx < 0 || idx >= batch.length - 2) {
      await loadBatch();
      idx = 0;
    }

    // Update feed with new post
    FEED.innerHTML = "";
    if (batch.length > 0) {
      FEED.appendChild(createPost(batch[idx]));

      // Start tracking time
      postViewStartTime = Date.now();
      lastViewedPostId = batch[idx].id;
    } else {
      // Handle empty batch
      const emptyDiv = document.createElement("div");
      emptyDiv.textContent = "No posts available.";
      FEED.appendChild(emptyDiv);

      postViewStartTime = null;
      lastViewedPostId = null;
    }

    // Update UI
    current = idx;
    debouncedRenderUI();
  } catch (error) {
    console.error("Error showing post:", error);
  }
}

// Navigation functions
function nextPost() {
  if (current < batch.length - 1) {
    showPost(current + 1);
  } else {
    loadBatch().then(() => showPost(0));
  }
}

function prevPost() {
  if (current > 0) {
    showPost(current - 1);
  }
}

// Keyboard navigation handler
function onKey(e) {
  if (e.key === "ArrowDown" || e.key === " ") {
    nextPost();
    e.preventDefault();
  } else if (e.key === "ArrowUp") {
    prevPost();
    e.preventDefault();
  }
}

// Handle visibility changes (pause/resume timers)
function handleVisibilityChange() {
  if (document.hidden) {
    // Page is hidden, record time and pause
    recordTimeSpentOnCurrentPost();
    postViewStartTime = null;
  } else {
    // Page is visible again, resume timer
    postViewStartTime = Date.now();
  }
}

// Render interaction history
function renderInteractionHistory() {
  try {
    // Sort interactions by time (most recent first)
    const allInteractions = interactions
      .slice()
      .sort((a, b) => b.timestamp - a.timestamp);

    let html = "<b>User Interactions</b><br>";
    html += '<ul style="margin:0 0 10px 0;padding-left:18px;overflow:auto;">';

    // Generate list items for each interaction
    for (const inter of allInteractions) {
      let details = [];
      if (inter.liked) details.push("👍");
      if (inter.interested) details.push("Interested");
      if (inter.not_interested) details.push("Not Interested");
      if (inter.commented) details.push("💬");
      if (inter.engagement) details.push("Engaged");

      let timeStr = `<div style='color:#0a7;font-size:0.95em;margin:2px 0 2px 0;'>
        <b>Time spent:</b> ${Math.round((inter.timeSpentMs || 0) / 1000)}s
      </div>`;

      html += `<li>
        <b>${details.join(", ") || "None"}</b> 
        <span style='color:#888'>[${inter.postId}]</span>${timeStr}
        <br>
        <span style='color:#555'>${(inter.topics || []).join(", ")}</span>
        <span style='color:#888'>${(inter.hashtags || []).join(", ")}</span>
      </li>`;
    }

    html += "</ul>";

    // // Add topics array info
    // html += "<b>Current Topics Array</b><br>";
    // html += `<div style='word-break:break-all;color:#1a4'>
    //   [${MODEL_TOPICS.map((t) => `'${t}'`).join(", ")}]
    // </div>`;

    INTERACTIONS_DIV.innerHTML = html;
  } catch (error) {
    console.warn("Error rendering interaction history:", error);
  }
}

// Render model status
async function renderModelStatus() {
  try {
    // Determine model status text and color
    let status = modelTraining
      ? '<span style="color:orange">Training</span>'
      : modelTrained
      ? '<span style="color:green">Trained</span>'
      : '<span style="color:red">Untrained</span>';

    let html = `<b>Model Status:</b> ${status}<br>`;

    // Add topics array
    // html += `<b>Topics Array</b><br>
    //   <div style='word-break:break-all;color:#1a4'>
    //     [${MODEL_TOPICS.map((t) => `'${t}'`).join(", ")}]
    //   </div>`;

    // Show most recent previous topics/hashtags instead of lastAnalyzed
    const latestTopics = previousTopicsResults.at(-1) || [];
    const latestHashtags = previousHashtagsResults.at(-1) || [];

    html += `<b>Next API Topics</b><br>
      <div style='word-break:break-all;color:#14a'>
        [${latestTopics
          .map(
            (t) =>
              `{ name: '${t.name}', weight: ${t.weight?.toFixed(3) ?? "?"} }`
          )
          .join(", ")}]
      </div>
      <b>Next API Hashtags</b><br>
      <div style='word-break:break-all;color:#14a'>
        [${latestHashtags
          .map(
            (h) =>
              `{ name: '${h.name}', weight: ${h.weight?.toFixed(3) ?? "?"} }`
          )
          .join(", ")}]
      </div>`;

    MODEL_STATUS_DIV.innerHTML = html;
  } catch (error) {
    console.warn("Error rendering model status:", error);
  }
}

// Debounced UI update to prevent excessive renders
function debouncedRenderUI() {
  if (uiUpdateDebounceTimer) {
    clearTimeout(uiUpdateDebounceTimer);
  }

  uiUpdateDebounceTimer = setTimeout(async () => {
    renderInteractionHistory();
    await renderModelStatus();
    uiUpdateDebounceTimer = null;
  }, UI_DEBOUNCE_TIME);
}

// Initialize when DOM is loaded
if (typeof window !== "undefined" && typeof document !== "undefined") {
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initializeApp);
  } else {
    initializeApp();
  }
}

// Export functions for testing or module usage
if (typeof module !== "undefined" && module.exports) {
  module.exports = {
    recordInteraction,
    analyzeInteractions,
    trainModel,
    nextPost,
    prevPost,
  };
}
