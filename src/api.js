// src/api.js
// Handles all API calls for posts and authentication
import { groth16 } from "snarkjs";

/**
 * Converts a hex string to a bit array of specified length
 * @param {string} hexStr - Hex string (with or without 0x prefix)
 * @param {number} bitLength - Length of the output bit array
 * @returns {Array<number>} - Array of bits (0s and 1s)
 */
function hexToBits(hexStr, bitLength = 256) {
  // Remove 0x prefix if present
  const cleanHex = hexStr.startsWith('0x') ? hexStr.slice(2) : hexStr;
  
  // Make sure we have enough characters to extract the requested bits
  const requiredHexChars = Math.ceil(bitLength / 4);
  const paddedHex = cleanHex.padStart(requiredHexChars, '0');
  
  // Convert to binary string
  let binStr = '';
  for (let i = 0; i < paddedHex.length; i++) {
    // Convert each hex character to 4 bits
    const hexChar = paddedHex[i];
    const nibble = parseInt(hexChar, 16);
    if (isNaN(nibble)) {
      throw new Error(`Invalid hex character: ${hexChar}`);
    }
    // Convert to 4-bit binary string and pad with leading zeros
    binStr += nibble.toString(2).padStart(4, '0');
  }
  
  // Trim or pad to exact length
  if (binStr.length > bitLength) {
    // Take only the last `bitLength` bits
    binStr = binStr.slice(-bitLength);
  } else if (binStr.length < bitLength) {
    // Pad with leading zeros
    binStr = binStr.padStart(bitLength, '0');
  }
  
  // Convert to array of 0s and 1s
  return Array.from(binStr).map(bit => parseInt(bit, 10));
}

/**
 * Generates a ZKP proof that the user is in the allowed list.
 */
export async function generateZKPProof(privKey, merkleRoot, pathElements, pathIndices) {
  const wasmPath = "/circuits/auth_js/auth.wasm";
  const zkeyPath = "/keys/circuit_final.zkey";

  try {
    console.log("Preparing ZKP inputs...");
    
    // Validate input formats
    if (!privKey || typeof privKey !== 'string') {
      throw new Error(`Invalid privKey format: ${privKey}`);
    }
    if (!merkleRoot || typeof merkleRoot !== 'string') {
      throw new Error(`Invalid merkleRoot format: ${merkleRoot}`);
    }
    if (!Array.isArray(pathElements) || pathElements.length === 0) {
      throw new Error(`Invalid pathElements: ${JSON.stringify(pathElements)}`);
    }
    if (!Array.isArray(pathIndices) || pathIndices.length === 0) {
      throw new Error(`Invalid pathIndices: ${JSON.stringify(pathIndices)}`);
    }
    
    // Ensure pathElements and pathIndices have the same length
    if (pathElements.length !== pathIndices.length) {
      throw new Error(`Path elements (${pathElements.length}) and indices (${pathIndices.length}) must have same length`);
    }

    // Convert inputs to bit arrays as expected by the circuit
    const privKeyBits = hexToBits(privKey, 256);
    const rootBits = hexToBits(merkleRoot, 256);
    
    // Convert path elements to bit arrays
    const pathBits = pathElements.map(element => hexToBits(element, 256));
    
    // Prepare input for the circuit
    const input = {
      leaf: privKeyBits,       // bit array of private key
      root: rootBits,          // bit array of merkle root
      path: pathBits,          // array of bit arrays
      direction: pathIndices   // array of 0s and 1s indicating left/right
    };

    console.log("Generating ZKP proof with formatted inputs...");
    
    const { proof, publicSignals } = await groth16.fullProve(
      input,
      wasmPath,
      zkeyPath
    );
    
    console.log("ZKP proof generated successfully");
    return { proof, publicSignals };
  } catch (error) {
    console.error("Error generating ZKP proof:", error);
    throw new Error(`ZKP proof generation failed: ${error.message}`);
  }
}

/**
 * Fetch a batch of posts from the backend API.
 * @param {Object} params - Query parameters (topics, hashtags, limit, etc.)
 * @param {Object} [auth] - Optional authentication object (e.g., ZKP proof)
 * @returns {Promise<Object>} - The API response JSON
 */
export async function fetchPosts(params = {}, auth = null) {
  const url = new URL("https://79a9-2404-c0-2120-00-1343-4b8.ap.ngrok.io/api/posts");
  Object.entries(params).forEach(([key, value]) => {
    if (Array.isArray(value)) {
      url.searchParams.append(key, value.join(","));
    } else {
      url.searchParams.append(key, value);
    }
  });

  const fetchOptions = {
    method: "GET",
    headers: {},
  };

  // If ZKP auth is provided, add it to headers or as needed
  if (auth && auth.proof) {
    fetchOptions.headers["X-ZKP-Proof"] = JSON.stringify(auth.proof);
    fetchOptions.headers["X-ZKP-PublicSignals"] = JSON.stringify(
      auth.publicSignals
    );
  }

  const res = await fetch(url.toString(), fetchOptions);
  if (!res.ok) throw new Error(`API error: ${res.status} ${res.statusText}`);
  return res.json();
}

// You can add more API functions here, e.g., for authentication, user profile, etc.