pragma circom 2.0.0;

include "../circomlib/circuits/sha256/sha256.circom";
include "merkle_tree_check.circom";

template AuthMerkleProof(nLevels) {
    signal input leaf[256];
    signal input root[256];
    signal input path[nLevels][256];
    signal input direction[nLevels];

    component merkleChecker = MerkleTreeCheck(nLevels);

    for (var j = 0; j < 256; j++) {
        merkleChecker.leaf[j] <== leaf[j];
        merkleChecker.root[j] <== root[j];
        for (var i = 0; i < nLevels; i++) {
            merkleChecker.path[i][j] <== path[i][j];
        }
    }
    for (var i = 0; i < nLevels; i++) {
        merkleChecker.direction[i] <== direction[i];
    }
}

component main = AuthMerkleProof(3); // 3-level tree: up to 8 users