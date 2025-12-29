#define FA_TILE_SIZE 16   // tile size for K/V blocking
#define FA_HEAD_DIM 64    // max head dimension for local memory

// forward pass
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

// forward pass with LSE output
__kernel void flash_attention_forward_with_lse(
    __global const float* Q,
    __global const float* K,
    __global const float* V,
    __global float* O,
    __global float* LSE,            // Log-Sum-Exp output [B, H, L]
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
    const int i = get_global_id(0);
    const int bh = get_global_id(1);

    if (i >= L || bh >= B * H) return;

    const int b = bh / H;
    const int h = bh - b * H;

    const int qBase = b * qStrides[0] + h * qStrides[1] + i * qStrides[2];
    const int oBase = b * oStrides[0] + h * oStrides[1] + i * oStrides[2];
    const int lseIdx = b * H * L + h * L + i;

    for (int d = 0; d < D; ++d) {
        O[oBase + d * oStrides[3]] = 0.0f;
    }

    float m = -INFINITY;
    float s = 0.0f;

    for (int j = 0; j < L; ++j) {
        if (causal && j > i) continue;

        const int kBase = b * kStrides[0] + h * kStrides[1] + j * kStrides[2];
        const int vBase = b * vStrides[0] + h * vStrides[1] + j * vStrides[2];

        float dot = 0.0f;
        for (int d = 0; d < D; ++d) {
            dot += Q[qBase + d * qStrides[3]] * K[kBase + d * kStrides[3]];
        }

        float score = dot * scale;
        float new_m = fmax(m, score);
        float e1 = ((isinf(m) && m < 0.0f) ? 0.0f : exp(m - new_m));
        float e2 = exp(score - new_m);

        for (int d = 0; d < D; ++d) {
            float old = O[oBase + d * oStrides[3]];
            float v   = V[vBase + d * vStrides[3]];
            O[oBase + d * oStrides[3]] = old * e1 + v * e2;
        }

        s = s * e1 + e2;
        m = new_m;
    }

    // store LSE = m + log(s) for backward pass
    LSE[lseIdx] = m + log(s + 1e-10f);

    float inv_s = (s > 0.0f) ? (1.0f / s) : 0.0f;
    for (int d = 0; d < D; ++d) {
        O[oBase + d * oStrides[3]] *= inv_s;
    }
}

// backward pass
__kernel void flash_attention_backward(
    __global const float* Q,
    __global const float* K,
    __global const float* V,
    __global const float* O,
    __global const float* dO,
    __global const float* LSE,
    __global float* dQ,
    __global float* dK,
    __global float* dV,
    __global const int* qStrides,
    __global const int* kStrides,
    __global const int* vStrides,
    __global const int* oStrides,
    __global const int* doStrides,
    __global const int* dkStrides,
    __global const int* dvStrides,
    const int B,
    const int H,
    const int L,
    const int D,
    const float scale,
    const int causal
) {
    const int j = get_global_id(0);  // key/value index
    const int bh = get_global_id(1);

    if (j >= L || bh >= B * H) return;

    const int b = bh / H;
    const int h = bh - b * H;

    const int kBase = b * kStrides[0] + h * kStrides[1] + j * kStrides[2];
    const int vBase = b * vStrides[0] + h * vStrides[1] + j * vStrides[2];
    const int dkBase = b * dkStrides[0] + h * dkStrides[1] + j * dkStrides[2];
    const int dvBase = b * dvStrides[0] + h * dvStrides[1] + j * dvStrides[2];

    // dK[j] and dV[j] accumulators inits
    float dK_acc[FA_HEAD_DIM];
    float dV_acc[FA_HEAD_DIM];
    for (int d = 0; d < D && d < FA_HEAD_DIM; ++d) {
        dK_acc[d] = 0.0f;
        dV_acc[d] = 0.0f;
    }

    // query position iteration
    for (int i = 0; i < L; ++i) {
        // causal mask: position i can only attend to j iff j <= i
        if (causal && j > i) continue;

        const int qBase = b * qStrides[0] + h * qStrides[1] + i * qStrides[2];
        const int oBase = b * oStrides[0] + h * oStrides[1] + i * oStrides[2];
        const int doBase = b * doStrides[0] + h * doStrides[1] + i * doStrides[2];
        const int lseIdx = b * H * L + h * L + i;

        // attention score recomputation: s_ij = Q[i] · K[j] * scale
        float score = 0.0f;
        for (int d = 0; d < D; ++d) {
            score += Q[qBase + d * qStrides[3]] * K[kBase + d * kStrides[3]];
        }
        score *= scale;

        // attention probability recomputation: p_ij = exp(s_ij - LSE[i])
        float lse_i = LSE[lseIdx];
        float p_ij = exp(score - lse_i);

        // D_i = sum_d(dO[i,d] * O[i,d])
        float D_i = 0.0f;
        for (int d = 0; d < D; ++d) {
            D_i += dO[doBase + d * doStrides[3]] * O[oBase + d * oStrides[3]];
        }

        // dP_ij = sum_d(dO[i,d] * V[j,d])
        float dP_ij = 0.0f;
        for (int d = 0; d < D; ++d) {
            dP_ij += dO[doBase + d * doStrides[3]] * V[vBase + d * vStrides[3]];
        }

        float dS_ij = p_ij * (dP_ij - D_i) * scale;

        for (int d = 0; d < D && d < FA_HEAD_DIM; ++d) {
            dK_acc[d] += dS_ij * Q[qBase + d * qStrides[3]];
            dV_acc[d] += p_ij * dO[doBase + d * doStrides[3]];
        }
    }

    for (int d = 0; d < D && d < FA_HEAD_DIM; ++d) {
        dK[dkBase + d * dkStrides[3]] = dK_acc[d];
        dV[dvBase + d * dvStrides[3]] = dV_acc[d];
    }
}

// backward pass for dQ only
__kernel void flash_attention_backward_dq(
    __global const float* Q,
    __global const float* K,
    __global const float* V,
    __global const float* O,
    __global const float* dO,
    __global const float* LSE,
    __global float* dQ,
    __global const int* qStrides,
    __global const int* kStrides,
    __global const int* vStrides,
    __global const int* oStrides,
    __global const int* doStrides,
    __global const int* dqStrides,
    const int B,
    const int H,
    const int L,
    const int D,
    const float scale,
    const int causal
) {
    const int i = get_global_id(0);  // query index
    const int bh = get_global_id(1);

    if (i >= L || bh >= B * H) return;

    const int b = bh / H;
    const int h = bh - b * H;

    const int qBase = b * qStrides[0] + h * qStrides[1] + i * qStrides[2];
    const int oBase = b * oStrides[0] + h * oStrides[1] + i * oStrides[2];
    const int doBase = b * doStrides[0] + h * doStrides[1] + i * doStrides[2];
    const int dqBase = b * dqStrides[0] + h * dqStrides[1] + i * dqStrides[2];
    const int lseIdx = b * H * L + h * L + i;
    float lse_i = LSE[lseIdx];

    // D_i = sum_d(dO[i,d] * O[i,d])
    float D_i = 0.0f;
    for (int d = 0; d < D; ++d) {
        D_i += dO[doBase + d * doStrides[3]] * O[oBase + d * oStrides[3]];
    }

    // dQ[i] accumulator init
    float dQ_acc[FA_HEAD_DIM];
    for (int d = 0; d < D && d < FA_HEAD_DIM; ++d) {
        dQ_acc[d] = 0.0f;
    }

    // key pos iteration
    int j_end = causal ? (i + 1) : L;
    for (int j = 0; j < j_end; ++j) {
        const int kBase = b * kStrides[0] + h * kStrides[1] + j * kStrides[2];
        const int vBase = b * vStrides[0] + h * vStrides[1] + j * vStrides[2];

        // attention score recomputation s_ij = Q[i] · K[j] * scale
        float score = 0.0f;
        for (int d = 0; d < D; ++d) {
            score += Q[qBase + d * qStrides[3]] * K[kBase + d * kStrides[3]];
        }
        score *= scale;

        // attention probability recomputation p_ij = exp(s_ij - LSE[i])
        float p_ij = exp(score - lse_i);

        // dP_ij = sum_d(dO[i,d] * V[j,d])
        float dP_ij = 0.0f;
        for (int d = 0; d < D; ++d) {
            dP_ij += dO[doBase + d * doStrides[3]] * V[vBase + d * vStrides[3]];
        }

        float dS_ij = p_ij * (dP_ij - D_i) * scale;

        // accumulate dQ[i] += dS_ij * K[j]
        for (int d = 0; d < D && d < FA_HEAD_DIM; ++d) {
            dQ_acc[d] += dS_ij * K[kBase + d * kStrides[3]];
        }
    }

    for (int d = 0; d < D && d < FA_HEAD_DIM; ++d) {
        dQ[dqBase + d * dqStrides[3]] = dQ_acc[d];
    }
}

// tiled forward pass with Local Memory Optimization.
// it requires FA_HEAD_DIM to be at least as large as D.
__kernel void flash_attention_forward_tiled(
    __global const float* Q,
    __global const float* K,
    __global const float* V,
    __global float* O,
    __global float* LSE,
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
    __local float K_tile[FA_TILE_SIZE * FA_HEAD_DIM];
    __local float V_tile[FA_TILE_SIZE * FA_HEAD_DIM];

    const int lid = get_local_id(0);
    const int wg_size = get_local_size(0);
    const int i = get_global_id(0);
    const int bh = get_global_id(1);

    if (bh >= B * H) return;

    const int b = bh / H;
    const int h = bh - b * H;

    // safety clamp for D
    const int D_clamped = (D < FA_HEAD_DIM) ? D : FA_HEAD_DIM;

    float m = -INFINITY;
    float s = 0.0f;
    float o_acc[FA_HEAD_DIM];

    if (i < L) {
        for (int d = 0; d < D_clamped; ++d) {
            o_acc[d] = 0.0f;
        }
    }

    const int qBase = (i < L) ? (b * qStrides[0] + h * qStrides[1] + i * qStrides[2]) : 0;

    // number of tiles
    const int numTiles = (L + FA_TILE_SIZE - 1) / FA_TILE_SIZE;

    for (int tile = 0; tile < numTiles; ++tile) {
        const int tile_start = tile * FA_TILE_SIZE;

        // coop loading of K and V tiles into local memory
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int t = lid; t < FA_TILE_SIZE; t += wg_size) {
            int j = tile_start + t;
            if (j < L) {
                const int kBase_t = b * kStrides[0] + h * kStrides[1] + j * kStrides[2];
                const int vBase_t = b * vStrides[0] + h * vStrides[1] + j * vStrides[2];
                for (int d = 0; d < D_clamped; ++d) {
                    K_tile[t * FA_HEAD_DIM + d] = K[kBase_t + d * kStrides[3]];
                    V_tile[t * FA_HEAD_DIM + d] = V[vBase_t + d * vStrides[3]];
                }
            } else {
                for (int d = 0; d < D_clamped; ++d) {
                    K_tile[t * FA_HEAD_DIM + d] = 0.0f;
                    V_tile[t * FA_HEAD_DIM + d] = 0.0f;
                }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        if (i >= L) continue;

        // tile processing
        for (int t = 0; t < FA_TILE_SIZE; ++t) {
            int j = tile_start + t;
            if (j >= L) break;
            if (causal && j > i) continue;

            // this computes the dot product Q[i] \cdot K[j] from the local memory
            float dot = 0.0f;
            for (int d = 0; d < D_clamped; ++d) {
                dot += Q[qBase + d * qStrides[3]] * K_tile[t * FA_HEAD_DIM + d];
            }

            float score = dot * scale;
            float new_m = fmax(m, score);
            float e1 = ((isinf(m) && m < 0.0f) ? 0.0f : exp(m - new_m));
            float e2 = exp(score - new_m);

            for (int d = 0; d < D_clamped; ++d) {
                o_acc[d] = o_acc[d] * e1 + V_tile[t * FA_HEAD_DIM + d] * e2;
            }

            s = s * e1 + e2;
            m = new_m;
        }
    }

    if (i >= L) return;

    // LSE storing and output normalization
    const int oBase = b * oStrides[0] + h * oStrides[1] + i * oStrides[2];
    const int lseIdx = b * H * L + h * L + i;

    LSE[lseIdx] = m + log(s + 1e-10f);

    float inv_s = (s > 0.0f) ? (1.0f / s) : 0.0f;
    for (int d = 0; d < D_clamped; ++d) {
        O[oBase + d * oStrides[3]] = o_acc[d] * inv_s;
    }
}

// tiled backward pass with Local Memory Optimization.
// it requires FA_HEAD_DIM to be at least as large as D.
__kernel void flash_attention_backward_tiled(
    __global const float* Q,
    __global const float* K,
    __global const float* V,
    __global const float* O,
    __global const float* dO,
    __global const float* LSE,
    __global float* dQ,
    __global float* dK,
    __global float* dV,
    __global const int* qStrides,
    __global const int* kStrides,
    __global const int* vStrides,
    __global const int* oStrides,
    __global const int* doStrides,
    __global const int* dkStrides,
    __global const int* dvStrides,
    const int B,
    const int H,
    const int L,
    const int D,
    const float scale,
    const int causal
) {
    __local float Q_tile[FA_TILE_SIZE * FA_HEAD_DIM];
    __local float dO_tile[FA_TILE_SIZE * FA_HEAD_DIM];
    __local float O_tile[FA_TILE_SIZE * FA_HEAD_DIM];

    const int lid = get_local_id(0);
    const int wg_size = get_local_size(0);
    const int j = get_global_id(0);
    const int bh = get_global_id(1);

    if (bh >= B * H) return;

    const int b = bh / H;
    const int h = bh - b * H;

    // accumulators for dK[j] and dV[j] inits
    float dK_acc[FA_HEAD_DIM];
    float dV_acc[FA_HEAD_DIM];

    if (j < L) {
        for (int d = 0; d < D && d < FA_HEAD_DIM; ++d) {
            dK_acc[d] = 0.0f;
            dV_acc[d] = 0.0f;
        }
    }

    const int kBase = (j < L) ? (b * kStrides[0] + h * kStrides[1] + j * kStrides[2]) : 0;
    const int vBase = (j < L) ? (b * vStrides[0] + h * vStrides[1] + j * vStrides[2]) : 0;
    const int dkBase = (j < L) ? (b * dkStrides[0] + h * dkStrides[1] + j * dkStrides[2]) : 0;
    const int dvBase = (j < L) ? (b * dvStrides[0] + h * dvStrides[1] + j * dvStrides[2]) : 0;

    const int numTiles = (L + FA_TILE_SIZE - 1) / FA_TILE_SIZE;

    for (int tile = 0; tile < numTiles; ++tile) {
        const int tile_start = tile * FA_TILE_SIZE;

        // coop loading of Q, O, dO tiles
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int t = lid; t < FA_TILE_SIZE; t += wg_size) {
            int i = tile_start + t;
            if (i < L) {
                const int qBase_t = b * qStrides[0] + h * qStrides[1] + i * qStrides[2];
                const int oBase_t = b * oStrides[0] + h * oStrides[1] + i * oStrides[2];
                const int doBase_t = b * doStrides[0] + h * doStrides[1] + i * doStrides[2];
                for (int d = 0; d < D; ++d) {
                    Q_tile[t * D + d] = Q[qBase_t + d * qStrides[3]];
                    dO_tile[t * D + d] = dO[doBase_t + d * doStrides[3]];
                    O_tile[t * D + d] = O[oBase_t + d * oStrides[3]];
                }
            } else {
                for (int d = 0; d < D; ++d) {
                    Q_tile[t * D + d] = 0.0f;
                    dO_tile[t * D + d] = 0.0f;
                    O_tile[t * D + d] = 0.0f;
                }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        if (j >= L) continue;

        // tile processing
        for (int t = 0; t < FA_TILE_SIZE; ++t) {
            int i = tile_start + t;
            if (i >= L) break;
            if (causal && j > i) continue;

            const int lseIdx = b * H * L + h * L + i;
            float lse_i = LSE[lseIdx];

            float score = 0.0f;
            for (int d = 0; d < D; ++d) {
                score += Q_tile[t * D + d] * K[kBase + d * kStrides[3]];
            }
            score *= scale;

            float p_ij = exp(score - lse_i);

            // compute D_i from tiled data
            float D_i = 0.0f;
            for (int d = 0; d < D; ++d) {
                D_i += dO_tile[t * D + d] * O_tile[t * D + d];
            }

            // then compute dP_ij
            float dP_ij = 0.0f;
            for (int d = 0; d < D; ++d) {
                dP_ij += dO_tile[t * D + d] * V[vBase + d * vStrides[3]];
            }

            float dS_ij = p_ij * (dP_ij - D_i) * scale;

            // gradients accumulation
            for (int d = 0; d < D && d < FA_HEAD_DIM; ++d) {
                dK_acc[d] += dS_ij * Q_tile[t * D + d];
                dV_acc[d] += p_ij * dO_tile[t * D + d];
            }
        }
    }

    if (j >= L) return;

    for (int d = 0; d < D && d < FA_HEAD_DIM; ++d) {
        dK[dkBase + d * dkStrides[3]] = dK_acc[d];
        dV[dvBase + d * dvStrides[3]] = dV_acc[d];
    }
}

