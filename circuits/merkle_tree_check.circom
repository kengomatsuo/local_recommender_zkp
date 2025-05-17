pragma circom 2.0.0;

// merkle_tree_check.circom

template MultiMux(n) {
    signal input sel;
    signal input in[2];
    signal output out;

    out <== in[0] * (1 - sel) + in[1] * sel;
}

template MerkleTreeCheck(nLevels) {
    signal input leaf[256]; // Now an array of bits
    signal input root[256]; // Now an array of bits
    signal input path[nLevels][256]; // Each path element is an array of bits
    signal input direction[nLevels];

    component concatLeft[nLevels];
    component concatRight[nLevels];
    component shaLeft[nLevels];
    component shaRight[nLevels];
    component shaFinal[nLevels];

    var current[256];
    for (var j = 0; j < 256; j++) current[j] = leaf[j];

    for (var i = 0; i < nLevels; i++) {
        // TODO: Concatenate current and path[i] as bits for Sha256 input
        // For now, just hash current || path[i]
        shaFinal[i] = Sha256(512);
        for (var j = 0; j < 256; j++) {
            shaFinal[i].in[j] <== current[j];
            shaFinal[i].in[256 + j] <== path[i][j];
        }
        for (var j = 0; j < 256; j++) current[j] = shaFinal[i].out[j];
    }

    for (var j = 0; j < 256; j++) {
        current[j] === root[j];
    }
}