package org.brain4j.math.gpu.ops;

import org.brain4j.math.gpu.GpuContext;
import org.brain4j.math.gpu.kernel.KernelFactory;
import org.brain4j.math.gpu.memory.GpuQueue;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.impl.GpuTensor;

public final class FlashAttention {

    private static final int FA_TILE_SIZE = 16;
    private static final int FA_HEAD_DIM = 64;

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

    /**
     * Computes the FlashAttention forward pass with Log-Sum-Exp (LSE) output.
     * This variant is required for backward pass computation.
     * <br>
     * <b>Parameters:</b>
     * <ul>
     *   <li><b>q</b> - Query tensor of shape [batch, heads, sequence_length, head_dim]. Must be a {@link GpuTensor}.</li>
     *   <li><b>k</b> - Key tensor of shape [batch, heads, sequence_length, head_dim]. Must be a {@link GpuTensor}.</li>
     *   <li><b>v</b> - Value tensor of shape [batch, heads, sequence_length, head_dim]. Must be a {@link GpuTensor}.</li>
     *   <li><b>scale</b> - Scaling factor applied to the attention scores (typically 1/sqrt(head_dim)).</li>
     *   <li><b>causal</b> - If {@code true}, applies causal masking.</li>
     * </ul>
     *
     * <b>Returns:</b>
     * <ul>
     *   <li>An array containing [O, LSE] where O is the attention output and LSE is the log-sum-exp values for backward.</li>
     *   <li>{@code null} if any input is not a {@link GpuTensor}.</li>
     * </ul>
     */
    public static Tensor[] forwardWithLse(Tensor q, Tensor k, Tensor v, double scale, boolean causal) {
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

        // LSE has shape [B, H, L]
        GpuTensor LSE = new GpuTensor(Q.device(), new int[]{B, H, L});

        long[] global = new long[] { L, (long) B * H };

        try (GpuQueue queue = GpuContext.getOrCreate(Q.device())) {
            KernelFactory.create(Q.device(), "flash_attention_forward_with_lse")
                .addMemParam(Q.dataBuffer())
                .addMemParam(K.dataBuffer())
                .addMemParam(V.dataBuffer())
                .addMemParam(O.dataBuffer())
                .addMemParam(LSE.dataBuffer())
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

        return new Tensor[] { O, LSE };
    }

    /**
     * Computes the FlashAttention backward pass on GPU tensors.
     * <br>
     * <b>Parameters:</b>
     * <ul>
     *   <li><b>q</b> - Query tensor of shape [batch, heads, sequence_length, head_dim].</li>
     *   <li><b>k</b> - Key tensor of shape [batch, heads, sequence_length, head_dim].</li>
     *   <li><b>v</b> - Value tensor of shape [batch, heads, sequence_length, head_dim].</li>
     *   <li><b>o</b> - Output tensor from forward pass.</li>
     *   <li><b>dO</b> - Gradient of the loss with respect to the output.</li>
     *   <li><b>lse</b> - Log-Sum-Exp values from forward pass with shape [batch, heads, sequence_length].</li>
     *   <li><b>scale</b> - Scaling factor (should match forward pass).</li>
     *   <li><b>causal</b> - If {@code true}, applies causal masking (should match forward pass).</li>
     * </ul>
     *
     * <b>Returns:</b>
     * <ul>
     *   <li>An array containing [dQ, dK, dV] gradients with same shapes as Q, K, V.</li>
     *   <li>{@code null} if any input is not a {@link GpuTensor} or inputs are on different devices.</li>
     * </ul>
     */
    public static Tensor[] backward(
            Tensor q,
            Tensor k,
            Tensor v,
            Tensor o,
            Tensor dO,
            Tensor lse,
            double scale,
            boolean causal
    ) {
        if (!(q instanceof GpuTensor Q) || !(k instanceof GpuTensor K) || !(v instanceof GpuTensor V) ||
            !(o instanceof GpuTensor O) || !(dO instanceof GpuTensor DO) || !(lse instanceof GpuTensor LSE)) {
            return null;
        }
        if (!Q.device().equals(K.device()) || !Q.device().equals(V.device()) ||
            !Q.device().equals(O.device()) || !Q.device().equals(DO.device()) || !Q.device().equals(LSE.device())) {
            return null;
        }

        int[] shape = Q.shape();
        int B = shape[0];
        int H = shape[1];
        int L = shape[2];
        int D = shape[3];

        // output gradient tensors setup
        GpuTensor dQ = new GpuTensor(Q.device(), shape);
        GpuTensor dK = new GpuTensor(Q.device(), shape);
        GpuTensor dV = new GpuTensor(Q.device(), shape);

        long[] global = new long[] { L, (long) B * H };

        try (GpuQueue queue = GpuContext.getOrCreate(Q.device())) {
            // first kernel: compute dK and dV
            KernelFactory.create(Q.device(), "flash_attention_backward")
                .addMemParam(Q.dataBuffer())
                .addMemParam(K.dataBuffer())
                .addMemParam(V.dataBuffer())
                .addMemParam(O.dataBuffer())
                .addMemParam(DO.dataBuffer())
                .addMemParam(LSE.dataBuffer())
                .addMemParam(dQ.dataBuffer())
                .addMemParam(dK.dataBuffer())
                .addMemParam(dV.dataBuffer())
                .addMemParam(Q.stridesBuffer())
                .addMemParam(K.stridesBuffer())
                .addMemParam(V.stridesBuffer())
                .addMemParam(O.stridesBuffer())
                .addMemParam(DO.stridesBuffer())
                .addMemParam(dK.stridesBuffer())
                .addMemParam(dV.stridesBuffer())
                .addIntParam(B)
                .addIntParam(H)
                .addIntParam(L)
                .addIntParam(D)
                .addFloatParam((float) scale)
                .addIntParam(causal ? 1 : 0)
                .launch(queue, 2, global);

            // second kernel: compute dQ
            KernelFactory.create(Q.device(), "flash_attention_backward_dq")
                .addMemParam(Q.dataBuffer())
                .addMemParam(K.dataBuffer())
                .addMemParam(V.dataBuffer())
                .addMemParam(O.dataBuffer())
                .addMemParam(DO.dataBuffer())
                .addMemParam(LSE.dataBuffer())
                .addMemParam(dQ.dataBuffer())
                .addMemParam(Q.stridesBuffer())
                .addMemParam(K.stridesBuffer())
                .addMemParam(V.stridesBuffer())
                .addMemParam(O.stridesBuffer())
                .addMemParam(DO.stridesBuffer())
                .addMemParam(dQ.stridesBuffer())
                .addIntParam(B)
                .addIntParam(H)
                .addIntParam(L)
                .addIntParam(D)
                .addFloatParam((float) scale)
                .addIntParam(causal ? 1 : 0)
                .launch(queue, 2, global);
        }

        return new Tensor[] { dQ, dK, dV };
    }

    /**
     * Computes the FlashAttention forward pass using tiled local memory optimization.
     * This version is more efficient for longer sequences.
     * <br>
     * <b>Parameters:</b>
     * <ul>
     *   <li><b>q</b> - Query tensor of shape [batch, heads, sequence_length, head_dim].</li>
     *   <li><b>k</b> - Key tensor of shape [batch, heads, sequence_length, head_dim].</li>
     *   <li><b>v</b> - Value tensor of shape [batch, heads, sequence_length, head_dim].</li>
     *   <li><b>scale</b> - Scaling factor applied to the attention scores.</li>
     *   <li><b>causal</b> - If {@code true}, applies causal masking.</li>
     * </ul>
     *
     * <b>Returns:</b>
     * <ul>
     *   <li>An array containing [O, LSE] where O is the attention output and LSE is the log-sum-exp values.</li>
     *   <li>{@code null} if any input is not a {@link GpuTensor}.</li>
     * </ul>
     */
    public static Tensor[] forwardTiled(Tensor q, Tensor k, Tensor v, double scale, boolean causal) {
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

        if (D > FA_HEAD_DIM) {
            // fall back to non-tiled version for large head dimensions
            return forwardWithLse(q, k, v, scale, causal);
        }

        GpuTensor O = new GpuTensor(Q.device(), shape);
        O.setAutogradContext(Q.autogradContext());

        GpuTensor LSE = new GpuTensor(Q.device(), new int[]{B, H, L});


        long[] global = new long[] { L, (long) B * H };
        long[] local = new long[] { Math.min(FA_TILE_SIZE, L), 1 };

        try (GpuQueue queue = GpuContext.getOrCreate(Q.device())) {
            KernelFactory.create(Q.device(), "flash_attention_forward_tiled")
                .addMemParam(Q.dataBuffer())
                .addMemParam(K.dataBuffer())
                .addMemParam(V.dataBuffer())
                .addMemParam(O.dataBuffer())
                .addMemParam(LSE.dataBuffer())
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
                .launch(queue, 2, global, local);
        }

        return new Tensor[] { O, LSE };
    }

    /**
     * Computes the FlashAttention backward pass using tiled local memory optimization.
     * <br>
     * <b>Parameters:</b>
     * <ul>
     *   <li><b>q</b> - Query tensor of shape [batch, heads, sequence_length, head_dim].</li>
     *   <li><b>k</b> - Key tensor of shape [batch, heads, sequence_length, head_dim].</li>
     *   <li><b>v</b> - Value tensor of shape [batch, heads, sequence_length, head_dim].</li>
     *   <li><b>o</b> - Output tensor from forward pass.</li>
     *   <li><b>dO</b> - Gradient of the loss with respect to the output.</li>
     *   <li><b>lse</b> - Log-Sum-Exp values from forward pass.</li>
     *   <li><b>scale</b> - Scaling factor (should match forward pass).</li>
     *   <li><b>causal</b> - If {@code true}, applies causal masking.</li>
     * </ul>
     *
     * <b>Returns:</b>
     * <ul>
     *   <li>An array containing [dQ, dK, dV] gradients.</li>
     *   <li>{@code null} if any input is not a {@link GpuTensor}.</li>
     * </ul>
     */
    public static Tensor[] backwardTiled(Tensor q, Tensor k, Tensor v, Tensor o, Tensor dO, Tensor lse,
                                         double scale, boolean causal) {
        if (!(q instanceof GpuTensor Q) || !(k instanceof GpuTensor K) || !(v instanceof GpuTensor V) ||
            !(o instanceof GpuTensor O) || !(dO instanceof GpuTensor DO) || !(lse instanceof GpuTensor LSE)) {
            return null;
        }
        if (!Q.device().equals(K.device()) || !Q.device().equals(V.device()) ||
            !Q.device().equals(O.device()) || !Q.device().equals(DO.device()) || !Q.device().equals(LSE.device())) {
            return null;
        }

        int[] shape = Q.shape();
        int B = shape[0];
        int H = shape[1];
        int L = shape[2];
        int D = shape[3];

        if (D > FA_HEAD_DIM) {
            // fall back to non-tiled version for large head dimensions
            return backward(q, k, v, o, dO, lse, scale, causal);
        }

        GpuTensor dQ = new GpuTensor(Q.device(), shape);
        GpuTensor dK = new GpuTensor(Q.device(), shape);
        GpuTensor dV = new GpuTensor(Q.device(), shape);

        long[] global = new long[] { L, (long) B * H };
        long[] local = new long[] { Math.min(FA_TILE_SIZE, L), 1 };

        try (GpuQueue queue = GpuContext.getOrCreate(Q.device())) {
            // tiled backward for dK and dV
            KernelFactory.create(Q.device(), "flash_attention_backward_tiled")
                .addMemParam(Q.dataBuffer())
                .addMemParam(K.dataBuffer())
                .addMemParam(V.dataBuffer())
                .addMemParam(O.dataBuffer())
                .addMemParam(DO.dataBuffer())
                .addMemParam(LSE.dataBuffer())
                .addMemParam(dQ.dataBuffer())
                .addMemParam(dK.dataBuffer())
                .addMemParam(dV.dataBuffer())
                .addMemParam(Q.stridesBuffer())
                .addMemParam(K.stridesBuffer())
                .addMemParam(V.stridesBuffer())
                .addMemParam(O.stridesBuffer())
                .addMemParam(DO.stridesBuffer())
                .addMemParam(dK.stridesBuffer())
                .addMemParam(dV.stridesBuffer())
                .addIntParam(B)
                .addIntParam(H)
                .addIntParam(L)
                .addIntParam(D)
                .addFloatParam((float) scale)
                .addIntParam(causal ? 1 : 0)
                .launch(queue, 2, global, local);

            // dQ kernel
            // TODO: optimize this with tiled version as well
            KernelFactory.create(Q.device(), "flash_attention_backward_dq")
                .addMemParam(Q.dataBuffer())
                .addMemParam(K.dataBuffer())
                .addMemParam(V.dataBuffer())
                .addMemParam(O.dataBuffer())
                .addMemParam(DO.dataBuffer())
                .addMemParam(LSE.dataBuffer())
                .addMemParam(dQ.dataBuffer())
                .addMemParam(Q.stridesBuffer())
                .addMemParam(K.stridesBuffer())
                .addMemParam(V.stridesBuffer())
                .addMemParam(O.stridesBuffer())
                .addMemParam(DO.stridesBuffer())
                .addMemParam(dQ.stridesBuffer())
                .addIntParam(B)
                .addIntParam(H)
                .addIntParam(L)
                .addIntParam(D)
                .addFloatParam((float) scale)
                .addIntParam(causal ? 1 : 0)
                .launch(queue, 2, global);
        }

        return new Tensor[] { dQ, dK, dV };
    }
}
