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
        boolean canFuse = (input instanceof GpuTensor) && !input.usesGrad();

        int batch = input.shape(0);
        int seq = input.shape(1);
        int d = headDimension;

        // project to QKV (no-grad if possible)
        Tensor QKV = canFuse ? input.matmul(qkvWeights) : input.matmulGrad(qkvWeights);

        Range all = Range.all();
        Tensor Q = QKV.slice(all, all, Range.interval(0, d));
        Tensor K = QKV.slice(all, all, Range.interval(d, 2 * d));
        Tensor V = QKV.slice(all, all, Range.interval(2 * d, 3 * d));

        if (canFuse) {
            // shapes to [B,1,L,d]
            Q = Q.reshape(batch, seq, d).unsqueeze(1);
            K = K.reshape(batch, seq, d).unsqueeze(1);
            V = V.reshape(batch, seq, d).unsqueeze(1);

            float scale = (float) (1.0 / Math.sqrt(d));
            Tensor context = FlashAttention.forward(Q, K, V, scale, false);
            if (context != null) {
                return context.squeeze(1); // [B,L,d]
            }
            // fallthrough if context null
        }

        // fallback to standard path with autograd support
        return super.attend(input);
    }
}
