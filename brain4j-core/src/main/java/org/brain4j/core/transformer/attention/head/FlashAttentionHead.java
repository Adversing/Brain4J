package org.brain4j.core.transformer.attention.head;

import org.brain4j.math.clipper.GradientClipper;
import org.brain4j.math.gpu.ops.FlashAttention;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.impl.GpuTensor;
import org.brain4j.math.tensor.index.Range;

public class FlashAttentionHead extends AttentionHead {

    public FlashAttentionHead(GradientClipper clipper, int embedDimension, int headDimension) {
        super(clipper, embedDimension, headDimension);
    }

    @Override
    public Tensor attend(Tensor input) {
        boolean isGpu = input instanceof GpuTensor;
        boolean training = input.usesGrad();

        int batch = input.shape(0);
        int seq = input.shape(1);
        int d = headDimension;

        // project to QKV
        Tensor QKV = training ? input.matmulGrad(qkvWeights) : input.matmul(qkvWeights);

        Range all = Range.all();
        Tensor Q, K, V;

        if (training) {
            Q = QKV.sliceGrad(all, all, Range.interval(0, d));
            K = QKV.sliceGrad(all, all, Range.interval(d, 2 * d));
            V = QKV.sliceGrad(all, all, Range.interval(2 * d, 3 * d));
        } else {
            Q = QKV.slice(all, all, Range.interval(0, d));
            K = QKV.slice(all, all, Range.interval(d, 2 * d));
            V = QKV.slice(all, all, Range.interval(2 * d, 3 * d));
        }

        if (isGpu) {
            // shapes to [B,1,L,d] for single-head attention
            if (training) {
                Q = Q.reshapeGrad(batch, seq, d).unsqueeze(1);
                K = K.reshapeGrad(batch, seq, d).unsqueeze(1);
                V = V.reshapeGrad(batch, seq, d).unsqueeze(1);
            } else {
                Q = Q.reshape(batch, seq, d).unsqueeze(1);
                K = K.reshape(batch, seq, d).unsqueeze(1);
                V = V.reshape(batch, seq, d).unsqueeze(1);
            }

            float scale = (float) (1.0 / Math.sqrt(d));

            Tensor context;
            if (training) {
                // use forward with LSE for training (required for backward pass)
                Tensor[] flashResult = FlashAttention.forwardWithLse(Q, K, V, scale, false);
                if (flashResult != null) {
                    context = flashResult[0];
                    // NOTE: LSE stored in flashResult[1] can be used if backward is needed
                } else {
                    context = null;
                }
            } else {
                context = FlashAttention.forward(Q, K, V, scale, false);
            }

            if (context != null) {
                return training
                    ? context.squeezeGrad(1) // [B,L,d]
                    : context.squeeze(1);
            }
            // fallthrough if context null
        }

        // fallback to standard path with autograd support
        return super.attend(input);
    }
}
