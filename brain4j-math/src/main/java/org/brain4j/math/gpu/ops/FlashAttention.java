package org.brain4j.math.gpu.ops;

import org.brain4j.math.gpu.GpuContext;
import org.brain4j.math.gpu.device.Device;
import org.brain4j.math.gpu.kernel.KernelFactory;
import org.brain4j.math.gpu.memory.GpuQueue;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.impl.GpuTensor;

public final class FlashAttention {

    private FlashAttention() { }

    /**
     * Computes the FlashAttention forward pass on GPU tensors.
     * <br>
     * <b>Parameters:</b>
     * <ul>
     *   <li><b>q</b> - Query tensor of shape [batch, heads, sequence_length, head_dim]. Must be a {@link GpuTensor}.</li>
     *   <li><b>k</b> - Key tensor of shape [batch, heads, sequence_length, head_dim]. Must be a {@link GpuTensor}.</li>
     *   <li><b>v</b> - Value tensor of shape [batch, heads, sequence_length, head_dim]. Must be a {@link GpuTensor}.</li>
     *   <li><b>scale</b> - Scaling factor applied to the attention scores (typically 1/sqrt(head_dim)).</li>
     *   <li><b>causal</b> - If {@code true}, applies causal masking so each position can only attend to previous positions (for autoregressive models).</li>
     * </ul>
     *
     * <b>Returns:</b>
     * <ul>
     *   <li>Output tensor of the same shape as {@code q} ([batch, heads, sequence_length, head_dim]) if all inputs are {@link GpuTensor}.</li>
     *   <li>{@code null} if any input is not a {@link GpuTensor}.</li>
     * </ul>
     *
     * <b>Behavior:</b>
     * <ul>
     *   <li>If any input tensor is not a {@link GpuTensor}, the method returns {@code null} and does not perform computation.</li>
     *   <li>All computation is performed on the GPU associated with the input tensors.</li>
     * </ul>
     */
    public static Tensor forward(Tensor q, Tensor k, Tensor v, double scale, boolean causal) {
        if (!(q instanceof GpuTensor Q) || !(k instanceof GpuTensor K) || !(v instanceof GpuTensor V)) {
                    return null;
        }
        if (!Q.device().equals(K.device()) || !Q.device().equals(V.device())) {
            return null;
        }

        int[] shape = Q.shape();
        int B = shape[0];
        int H = shape[1];
        int L = shape[2];
        int D = shape[3];

        GpuTensor O = new GpuTensor(Q.device(), shape);
        O.setAutogradContext(Q.autogradContext());

        long[] global = new long[] { L, (long) B * H };

        try (GpuQueue queue = GpuContext.getOrCreate(Q.device())) {
            KernelFactory.create(Q.device(), "flash_attention_forward")
                .addMemParam(Q.dataBuffer())
                .addMemParam(K.dataBuffer())
                .addMemParam(V.dataBuffer())
                .addMemParam(O.dataBuffer())
                .addMemParam(Q.stridesBuffer())
                .addMemParam(K.stridesBuffer())
                .addMemParam(V.stridesBuffer())
                .addMemParam(O.stridesBuffer())
                .addIntParam(B)
                .addIntParam(H)
                .addIntParam(L)
                .addIntParam(D)
                .addFloatParam((float) scale)
                .addIntParam(causal ? 1 : 0)
                .launch(queue, 2, global);
        }

        return O;
    }
}
