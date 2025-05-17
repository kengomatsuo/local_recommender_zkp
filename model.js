// model.js
// Model logic separated from main.js

// ML Model State
export let tfModel = null;
export let modelTrained = false;
export let modelTraining = false;
export let modelTrainingPromise = null;
export let MODEL_TOPICS = [];
export let MODEL_HASHTAGS = [];

// Preference weighting constants
export const WEIGHT_LIKED = 3.0;
export const WEIGHT_INTERESTED = 2.0;
export const WEIGHT_NOT_INTERESTED = -4.0;
export const WEIGHT_COMMENTED = 1.5;
export const MIN_INTERACTIONS = 10;

// Load model with improved architecture
export async function loadModel(force = false) {
  try {
    if (modelTraining && modelTrainingPromise) {
      await modelTrainingPromise;
    }
    if (!tfModel || force) {
      if (MODEL_TOPICS.length === 0) {
        tfModel = null;
        modelTrained = false;
        return;
      }
      if (tfModel) {
        try { tfModel.dispose(); } catch (e) { }
      }
      tfModel = tf.sequential();
      tfModel.add(tf.layers.dense({
        inputShape: [MODEL_TOPICS.length],
        units: Math.max(16, MODEL_TOPICS.length),
        activation: "relu",
        kernelRegularizer: tf.regularizers.l2(0.001)
      }));
      tfModel.add(tf.layers.dense({
        units: 8,
        activation: "relu",
        kernelRegularizer: tf.regularizers.l2(0.001)
      }));
      tfModel.add(tf.layers.dense({
        units: 3,
        activation: "softmax"
      }));
      const optimizer = tf.train.adam(0.001);
      tfModel.compile({
        optimizer: optimizer,
        loss: "categoricalCrossentropy",
        metrics: ["accuracy"]
      });
      modelTrained = false;
    }
  } catch (error) {
    modelTrained = false;
  }
}

// Train model
export async function trainModel(interactionsArr) {
  if (modelTraining || MODEL_TOPICS.length === 0) return;
  const interactions = interactionsArr;
  if (interactions.length < MIN_INTERACTIONS) return;
  let resolveTraining;
  try {
    await loadModel(true);
    modelTraining = true;
    modelTrainingPromise = new Promise((r) => (resolveTraining = r));
    const recentInteractions = interactions.slice(-100);
    const xs = [], ys = [];
    for (const inter of recentInteractions) {
      if (!inter.topics || !Array.isArray(inter.topics)) continue;
      let preferenceScore = 0;
      if (inter.liked) preferenceScore += WEIGHT_LIKED;
      if (inter.interested) preferenceScore += WEIGHT_INTERESTED;
      if (inter.not_interested) preferenceScore += WEIGHT_NOT_INTERESTED;
      if (inter.commented) preferenceScore += WEIGHT_COMMENTED;
      if (!inter.liked && !inter.interested && !inter.not_interested && !inter.commented) {
        const time_watched = inter.timeSpentMs || 0;
        const duration = inter.duration || 10000;
        const timeRatio = time_watched / duration;
        const timeScore = (timeRatio - 0.5) * 2;
        preferenceScore += timeScore;
      }
      let engaged;
      if (preferenceScore <= -1.5) engaged = 0;
      else if (preferenceScore >= 1.5) engaged = 2;
      else engaged = 1;
      const x = MODEL_TOPICS.map((t) => (inter.topics || []).includes(t) ? 1 : 0);
      const y = [0, 0, 0];
      y[engaged] = 1;
      xs.push(x);
      ys.push(y);
    }
    if (xs.length > 0 && xs[0].length > 0) {
      const xsTensor = tf.tensor2d(xs);
      const ysTensor = tf.tensor2d(ys);
      try {
        await tfModel.fit(xsTensor, ysTensor, {
          epochs: 25,
          batchSize: 8,
          validationSplit: 0.2,
        });
        modelTrained = true;
      } finally {
        xsTensor.dispose();
        ysTensor.dispose();
      }
    }
  } catch (error) {
    modelTrained = false;
  } finally {
    modelTraining = false;
    if (resolveTraining) resolveTraining();
    modelTrainingPromise = null;
  }
}

// Analyze interactions
export async function analyzeInteractions(interactionsArr) {
  const interactions = interactionsArr;
  if (MODEL_TOPICS.length === 0) return { topics: [], hashtags: [] };
  if (interactions.length < MIN_INTERACTIONS) return { topics: [], hashtags: [] };
  if (!modelTrained) return analyzeWithoutModel(interactions);
  await loadModel(false);
  const topicInputs = MODEL_TOPICS.map((t, i) =>
    MODEL_TOPICS.map((_, j) => (i === j ? 1 : 0))
  );
  const topicTensor = tf.tensor2d(topicInputs);
  const topicPreds = tfModel.predict(topicTensor).arraySync();
  topicTensor.dispose();
  const scoredTopics = MODEL_TOPICS.map((t, i) => ({
    name: t,
    weight: topicPreds[i][2] - topicPreds[i][0],
  }));
  scoredTopics.sort((a, b) => b.weight - a.weight);
  const splitIdx = findNaturalSplit(scoredTopics, 0.1, 'weight');
  const topTopics = scoredTopics.slice(0, splitIdx).filter((e) => e.weight > 0.1);
  // Hashtag analysis
  const hashtagInputs = MODEL_HASHTAGS.map((h, i) =>
    MODEL_TOPICS.map((t) =>
      interactions.some(
        (inter) => (inter.hashtags || []).includes(h) && (inter.topics || []).includes(t)
      ) ? 1 : 0
    )
  );
  const hashtagTensor = tf.tensor2d(hashtagInputs);
  const hashtagPreds = tfModel.predict(hashtagTensor).arraySync();
  hashtagTensor.dispose();
  const scoredHashtags = MODEL_HASHTAGS.map((h, i) => ({
    name: h,
    weight: hashtagPreds[i][2] - hashtagPreds[i][0],
  }));
  scoredHashtags.sort((a, b) => b.weight - a.weight);
  const splitHIdx = findNaturalSplit(scoredHashtags, 0.1, 'weight');
  const topHashtags = scoredHashtags.slice(0, splitHIdx).filter((e) => e.weight > 0.1);
  return { topics: topTopics, hashtags: topHashtags };
}

// Simple analysis method when model isn't ready
export function analyzeWithoutModel(interactions) {
  const topicScores = {};
  const hashtagScores = {};
  MODEL_TOPICS.forEach(topic => {
    topicScores[topic] = { positive: 0, negative: 0, count: 0 };
  });
  MODEL_HASHTAGS.forEach(hashtag => {
    hashtagScores[hashtag] = { positive: 0, negative: 0, count: 0 };
  });
  for (const inter of interactions) {
    let score = 0;
    if (inter.liked) score += WEIGHT_LIKED;
    if (inter.interested) score += WEIGHT_INTERESTED;
    if (inter.not_interested) score += WEIGHT_NOT_INTERESTED;
    if (inter.commented) score += WEIGHT_COMMENTED;
    const timeRatio = inter.timeSpentMs / (inter.duration || 10000);
    if (timeRatio > 0.7) score += 1;
    if (timeRatio < 0.2) score -= 1;
    (inter.topics || []).forEach(topic => {
      if (topicScores[topic]) {
        if (score > 0) topicScores[topic].positive += score;
        if (score < 0) topicScores[topic].negative += Math.abs(score);
        topicScores[topic].count++;
      }
    });
    (inter.hashtags || []).forEach(hashtag => {
      if (hashtagScores[hashtag]) {
        if (score > 0) hashtagScores[hashtag].positive += score;
        if (score < 0) hashtagScores[hashtag].negative += Math.abs(score);
        hashtagScores[hashtag].count++;
      }
    });
  }
  const scoredTopics = Object.entries(topicScores)
    .map(([topic, data]) => {
      const avg = data.count > 0 ? (data.positive - data.negative) / data.count : 0;
      const total = data.positive - data.negative;
      // Composite: 60% average, 40% total (normalize total by dividing by max total)
      return {
        name: topic,
        weight: avg * 0.6 + (total / Math.max(1, interactions.length)) * 0.4
      };
    })
    .filter(item => item.weight !== 0)
    .sort((a, b) => b.weight - a.weight);

  const scoredHashtags = Object.entries(hashtagScores)
    .map(([hashtag, data]) => {
      const avg = data.count > 0 ? (data.positive - data.negative) / data.count : 0;
      const total = data.positive - data.negative;
      return {
        name: hashtag,
        weight: avg * 0.6 + (total / Math.max(1, interactions.length)) * 0.4
      };
    })
    .filter(item => item.weight !== 0)
    .sort((a, b) => b.weight - a.weight);

  const topTopics = scoredTopics.filter(item => item.weight > 0.1).slice(0, 5);
  const topHashtags = scoredHashtags.filter(item => item.weight > 0.1).slice(0, 5);
  return { topics: topTopics, hashtags: topHashtags };
}

// Helper to find natural split
export function findNaturalSplit(items, minThreshold = 0.1, scoreKey = "score") {
  if (items.length <= 3) return items.length;
  let splitIdx = items.length;
  let maxGap = 0;
  let gapThreshold = Math.max(0.1, items[0][scoreKey] * 0.25);
  for (let i = 0; i < Math.min(items.length - 1, 10); i++) {
    const gap = items[i][scoreKey] - items[i + 1][scoreKey];
    const relativeGap = gap / items[i][scoreKey];
    if ((gap > maxGap && gap > gapThreshold) || relativeGap > 0.4) {
      maxGap = gap;
      splitIdx = i + 1;
      if (relativeGap > 0.6) break;
    }
  }
  if (splitIdx === items.length) {
    return Math.min(5, items.filter(item => item[scoreKey] > minThreshold).length);
  }
  return splitIdx;
}
