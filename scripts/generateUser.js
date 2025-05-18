import SHA256 from "crypto-js/sha256.js";
import { MerkleTree } from "merkletreejs";

const users = Array.from({ length: 8 }, (_, i) => `user${i}-secret`);
const leaves = users.map(u => SHA256(u).toString());
const tree = new MerkleTree(leaves, SHA256);
const root = tree.getRoot().toString("hex");

// Pick a user to prove
const index = 0; // pick the first user
const leaf = leaves[index];
const proof = tree.getProof(leaf);
const hexProof = proof.map(p => p.data.toString("hex"));
const directions = proof.map(p => (p.position === "left" ? 0 : 1));

console.log("Merkle Root:", root);
console.log("Example Leaf:", leaf.toString("hex"));
console.log("Hex Path:", hexProof);
console.log("Directions:", directions);