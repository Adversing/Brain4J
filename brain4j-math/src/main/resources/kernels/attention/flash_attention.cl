__kernel void flash_attention_forward(
    __global const float* Q,
    __global const float* K,
    __global const float* V,
    __global float* O,
    __global const int* qStrides,
    __global const int* kStrides,
    __global const int* vStrides,
    __global const int* oStrides,
    const int B,
    const int H,
    const int L,
    const int D,
    const float scale,
    const int causal
) {
    // global work layout: [L, B*H]
    const int i = get_global_id(0);  // query index in sequence [0..L)
    const int bh = get_global_id(1); // combined batch*head index [0..B*H)

    if (i >= L || bh >= B * H) return;

    // batch and head indices decoding
    const int b = bh / H;
    const int h = bh - b * H;

    // calc of base offsets using strides
    const int qBase = b * qStrides[0] + h * qStrides[1] + i * qStrides[2];
    const int oBase = b * oStrides[0] + h * oStrides[1] + i * oStrides[2];

    // output accumulator y (numerator of softmax) to 0 init
    for (int d = 0; d < D; ++d) {
        O[oBase + d * oStrides[3]] = 0.0f;
    }

    // default softmax state
    float m = -INFINITY; // running max
    float s = 0.0f;      // running sum

    // k-v iteration
    for (int j = 0; j < L; ++j) {
        if (causal && j > i) continue; // causal mask

        const int kBase = b * kStrides[0] + h * kStrides[1] + j * kStrides[2];
        const int vBase = b * vStrides[0] + h * vStrides[1] + j * vStrides[2];

        // dot(Q[i], K[j])
        float dot = 0.0f;
        for (int d = 0; d < D; ++d) {
            dot += Q[qBase + d * qStrides[3]] * K[kBase + d * kStrides[3]];
        }

        float score = dot * scale;
        float new_m = fmax(m, score);
        float e1 = ((isinf(m) && m < 0.0f) ? 0.0f : exp(m - new_m));
        float e2 = exp(score - new_m);

        // numerator accumulator update to vector O = O * e1 + V[j] * e2
        for (int d = 0; d < D; ++d) {
            float old = O[oBase + d * oStrides[3]];
            float v   = V[vBase + d * vStrides[3]];
            O[oBase + d * oStrides[3]] = old * e1 + v * e2;
        }

        s = s * e1 + e2;
        m = new_m;
    }

    // output normalization
    float inv_s = (s > 0.0f) ? (1.0f / s) : 0.0f;
    for (int d = 0; d < D; ++d) {
        O[oBase + d * oStrides[3]] *= inv_s;
    }
}
