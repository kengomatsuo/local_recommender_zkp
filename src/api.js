// src/api.js
// Handles all API calls for posts and authentication (ZKP ready)

/**
 * Fetch a batch of posts from the backend API.
 * @param {Object} params - Query parameters (topics, hashtags, limit, etc.)
 * @param {Object} [auth] - Optional authentication object (e.g., ZKP proof)
 * @returns {Promise<Object>} - The API response JSON
 */
export async function fetchPosts(params = {}, auth = null) {
  const url = new URL('http://localhost:3000/api/posts');
  Object.entries(params).forEach(([key, value]) => {
    if (Array.isArray(value)) {
      url.searchParams.append(key, value.join(','));
    } else {
      url.searchParams.append(key, value);
    }
  });

  const fetchOptions = {
    method: 'GET',
    headers: {}
  };

  // If ZKP auth is provided, add it to headers or as needed
  if (auth && auth.proof) {
    fetchOptions.headers['X-ZKP-Proof'] = auth.proof;
  }

  const res = await fetch(url.toString(), fetchOptions);
  if (!res.ok) throw new Error(`API error: ${res.status} ${res.statusText}`);
  return res.json();
}

// You can add more API functions here, e.g., for authentication, user profile, etc.
