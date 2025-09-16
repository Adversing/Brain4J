#define TILE_SIZE 16

__kernel void matmul_batched(
    __global const float* A,
    __global const float* B,
    __global float* C,
    __global const int* offsetsA,
    __global const int* offsetsB,
    __global const int* offsetsC,
    const int M,
    const int N,
    const int P,
    const int batchCount
) {
    int row = get_global_id(0);
    int col = get_global_id(1);
    int batch = get_global_id(2);

    int local_row = get_local_id(0);
    int local_col = get_local_id(1);

    if (batch >= batchCount || row >= M || col >= P) return;

    const __global float* A_batch = A + offsetsA[batch];
    const __global float* B_batch = B + offsetsB[batch];
    __global float* C_batch = C + offsetsC[batch];

    __local float Asub[TILE_SIZE][TILE_SIZE];
    __local float Bsub[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;

    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; ++t) {
        int tiled_col_A = t * TILE_SIZE + local_col;
        int tiled_row_B = t * TILE_SIZE + local_row;

        if (row < M && tiled_col_A < N) {
            Asub[local_row][local_col] = A_batch[row * N + tiled_col_A];
        } else {
            Asub[local_row][local_col] = 0.0f;
        }

        if (tiled_row_B < N && col < P) {
            Bsub[local_row][local_col] = B_batch[tiled_row_B * P + col];
        } else {
            Bsub[local_row][local_col] = 0.0f;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += Asub[local_row][k] * Bsub[k][local_col];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    C_batch[row * P + col] = sum;
}

__kernel void matmul(
    __global float* A,
    __global float* B,
    __global float* C,
    const int M,
    const int N,
    const int P
) {
    __local float Asub[TILE_SIZE][TILE_SIZE];
    __local float Bsub[TILE_SIZE][TILE_SIZE];

    int row = get_global_id(0);
    int col = get_global_id(1);
    int local_row = get_local_id(0);
    int local_col = get_local_id(1);

    float sum = 0.0f;

    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        int tiled_row = row;
        int tiled_col = t * TILE_SIZE + local_col;
        if (tiled_row < M && tiled_col < N)
            Asub[local_row][local_col] = A[tiled_row * N + tiled_col];
        else
            Asub[local_row][local_col] = 0.0f;

        tiled_row = t * TILE_SIZE + local_row;
        tiled_col = col;
        if (tiled_row < N && tiled_col < P)
            Bsub[local_row][local_col] = B[tiled_row * P + tiled_col];
        else
            Bsub[local_row][local_col] = 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += Asub[local_row][k] * Bsub[k][local_col];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (row < M && col < P)
        C[row * P + col] = sum;
}

__kernel void add(
    __global float* a,
    __global const float* b,
    int size,
    int broadcast_dim,
    int batch
) {
    int gid = get_global_id(0);

    if (broadcast_dim == -1) {
        if (gid < size) {
            a[gid] += b[gid];
        }
    } else {
        if (gid < size) {
            int j = gid % broadcast_dim;
            a[gid] += b[j];
        }
    }
}

__kernel void sub(
    __global float* a,
    __global const float* b,
    int size,
    int broadcast_dim,
    int batch
) {
    int gid = get_global_id(0);

    if (broadcast_dim == -1) {
        if (gid < size) {
            a[gid] -= b[gid];
        }
    } else {
        if (gid < size) {
            int j = gid % broadcast_dim;
            a[gid] -= b[j];
        }
    }
}

__kernel void mul(
    __global float* a,
    __global const float* b,
    int size,
    int broadcast_dim,
    int batch
) {
    int gid = get_global_id(0);

    if (broadcast_dim == -1) {
        if (gid < size) {
            a[gid] *= b[gid];
        }
    } else {
        if (gid < size) {
            int j = gid % broadcast_dim;
            a[gid] *= b[j];
        }
    }
}

__kernel void div(
    __global float* a,
    __global const float* b,
    int size,
    int broadcast_dim,
    int batch
) {
    int gid = get_global_id(0);

    if (broadcast_dim == -1) {
        if (gid < size) {
            a[gid] /= b[gid];
        }
    } else {
        if (gid < size) {
            int j = gid % broadcast_dim;
            a[gid] /= b[j];
        }
    }
}

__kernel void transpose(
    __global const float* input,
    __global float* output,
    __global const int* srcStrides,
    __global const int* dstStrides,
    const int rank
) {
    int dstLinearIdx = get_global_id(0);
    int dstOffset = 0;
    int srcOffset = 0;
    int tmp = dstLinearIdx;

    // invece di un array idx, usiamo due variabili per le ultime due dimensioni
    int idx_second_last = 0;
    int idx_last = 0;

    for (int i = 0; i < rank; i++) {
        int idx_i = tmp / dstStrides[i];
        tmp = tmp % dstStrides[i];
        dstOffset += idx_i * dstStrides[i];

        if (i == rank-2) idx_second_last = idx_i;
        else if (i == rank-1) idx_last = idx_i;
    }

    tmp = dstLinearIdx;
    for (int i = 0; i < rank; i++) {
        int srcIdx;
        int idx_i = tmp / dstStrides[i]; // o calcolato separatamente
        tmp = tmp % dstStrides[i];

        if (i == rank-2) srcIdx = idx_last;
        else if (i == rank-1) srcIdx = idx_second_last;
        else srcIdx = idx_i;

        srcOffset += srcIdx * srcStrides[i];
    }
    output[dstOffset] = input[srcOffset];
}


__kernel void sum_along_dim(
    __global const float* input,
    __global float* output,
    const int outerSize,
    const int reducedSize,
    const int innerSize
) {
    int gid_outer = get_global_id(0);
    int gid_inner = get_global_id(1);

    if (gid_outer >= outerSize || gid_inner >= innerSize) return;

    float sum = 0.0f;
    for (int i = 0; i < reducedSize; i++) {
        int idx = gid_outer * reducedSize * innerSize + i * innerSize + gid_inner;
        sum += input[idx];
    }

    int resultIndex = gid_outer * innerSize + gid_inner;
    output[resultIndex] = sum;
}

__kernel void layer_norm(
    __global float* data,
    const int batchSize,
    const int featuresSize,
    const float epsilon
) {
    int batch_idx = get_global_id(0);

    if (batch_idx >= batchSize) return;

    int offset = batch_idx * featuresSize;

    float mean = 0.0f;

    for (int j = 0; j < featuresSize; j++) {
        mean += data[offset + j];
    }

    mean /= featuresSize;

    float variance = 0.0f;

    for (int j = 0; j < featuresSize; j++) {
        float diff = data[offset + j] - mean;
        variance += diff * diff;
    }

    variance /= featuresSize;

    float denom = sqrt(variance + epsilon);

    for (int j = 0; j < featuresSize; j++) {
        data[offset + j] = (data[offset + j] - mean) / denom;
    }
}

__kernel void softmax_last_dim(
    __global const float* input,
    __global float* output,
    const int lastDim,
    const float temperature
) {
    int row = get_global_id(0);

    int offset = row * lastDim;

    float max_val = input[offset];
    for (int i = 1; i < lastDim; i++) {
        float val = input[offset + i];
        if (val > max_val) max_val = val;
    }

    float sum = 0.0f;
    for (int i = 0; i < lastDim; i++) {
        float val = (input[offset + i] - max_val) / temperature;
        float exp_val = exp(val);
        output[offset + i] = exp_val;
        sum += exp_val;
    }

    for (int i = 0; i < lastDim; i++) {
        output[offset + i] /= sum;
    }
}

__kernel void slice(
    __global const float* srcData,
    __global float* dstData,
    __global const int* srcStrides,
    __global const int* dstStrides,
    __global const int* dstShape,
    __global const int* starts,
    __global const int* steps,
    const int rank
) {
    int dstLinearIdx = get_global_id(0);

    int totalElements = 1;
    for (int i = 0; i < rank; i++) totalElements *= dstShape[i];
    if (dstLinearIdx >= totalElements) return;

    int tmp = dstLinearIdx;
    int srcOffset = 0;

    for (int i = 0; i < rank; i++) {
        int idx = tmp / dstStrides[i];
        tmp = tmp % dstStrides[i];

        int srcIdx = starts[i] + idx * steps[i];
        srcOffset += srcIdx * srcStrides[i];
    }

    dstData[dstLinearIdx] = srcData[srcOffset];
}

__kernel void concat_last_dim(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const int outerSize,
    const int lastA,
    const int lastB,
    const int concatLast
) {
    int gid = get_global_id(0);
    int total = outerSize * concatLast;
    if (gid >= total) return;

    int row = gid / concatLast;
    int col = gid % concatLast;

    if (col < lastA) {
        int aIdx = row * lastA + col;
        C[gid] = A[aIdx];
    } else {
        int bIdx = row * lastB + (col - lastA);
        C[gid] = B[bIdx];
    }
}

__kernel void concat_copy_a(
    __global const float* A,
    __global float* C,
    const int numBlocks,
    const int thisDim,
    const int otherDim,
    const int blockSize
) {
    int gid = get_global_id(0);
    int blockElem = thisDim * blockSize;
    int total = numBlocks * blockElem;
    if (gid >= total) return;

    int block = gid / blockElem;
    int inBlockIdx = gid % blockElem;

    int destStride = (thisDim + otherDim) * blockSize;
    int destIndex = block * destStride + inBlockIdx;

    C[destIndex] = A[gid];
}

__kernel void concat_copy_b(
    __global const float* B,
    __global float* C,
    const int numBlocks,
    const int thisDim,
    const int otherDim,
    const int blockSize
) {
    int gid = get_global_id(0);
    int blockElem = otherDim * blockSize;
    int total = numBlocks * blockElem;
    if (gid >= total) return;

    int block = gid / blockElem;
    int inBlockIdx = gid % blockElem;

    int destStride = (thisDim + otherDim) * blockSize;
    int destIndex = block * destStride + thisDim * blockSize + inBlockIdx;

    C[destIndex] = B[gid];
}
