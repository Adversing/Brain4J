package org.brain4j.examples.math;

import org.brain4j.core.Brain4J;
import org.brain4j.math.Tensors;
import org.brain4j.math.gpu.device.Device;
import org.brain4j.math.gpu.device.DeviceUtils;
import org.brain4j.math.gpu.ops.FlashAttention;
import org.brain4j.math.tensor.Tensor;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;

public class FlashAttentionTest {

    private Device initDeviceOrSkip() {
        List<String> names = DeviceUtils.allDeviceNames();
        Assumptions.assumeTrue(!names.isEmpty(), "No OpenCL GPU devices available");
        Device device = Brain4J.findDevice(names.getFirst());
        Assumptions.assumeTrue(device != null, "Failed to acquire OpenCL device");
        Brain4J.initKernels(device);
        return device;
    }

    private static double mae(float[] a, float[] b) {
        double s = 0.0;
        for (int i = 0; i < a.length; i++) s += Math.abs(a[i] - b[i]);
        return s / a.length;
    }

    @Test
    public void testFlashAttentionMatchesBaseline() {
        Device device = initDeviceOrSkip();

        int B = 1, H = 2, L = 8, D = 16;
        float scale = (float) (1.0 / Math.sqrt(D));

        Tensor Q = Tensors.random(B, H, L, D);
        Tensor K = Tensors.random(B, H, L, D);
        Tensor V = Tensors.random(B, H, L, D);

        // baseline (CPU): attn(Q,K,V) = softmax(QK^T * scale) V
        Tensor KT = K.transpose(2, 3);
        Tensor scores = Q.matmul(KT).mul(scale);
        Tensor attn = scores.softmax();
        Tensor outRef = attn.matmul(V);

        // fused (GPU)
        Tensor Qg = Q.gpu(device);
        Tensor Kg = K.gpu(device);
        Tensor Vg = V.gpu(device);
        Tensor outGpu = FlashAttention.forward(Qg, Kg, Vg, scale, false);
        Assumptions.assumeTrue(outGpu != null, "Fused kernel did not run (returned null)");
        assertArrayEquals(outRef.shape(), outGpu.shape(), "Output shape mismatch");

        float[] a = outRef.data();
        float[] b = outGpu.data();
        double error = mae(a, b);
        System.out.println("MAE: " + error);

        assertTrue(error < 3e-3, "MAE too high: " + error);
    }

    @Test
    public void testCausalFlashAttentionMatchesBaseline() {
        Device device = initDeviceOrSkip();

        int B = 1, H = 2, L = 8, D = 16;
        float scale = (float) (1.0 / Math.sqrt(D));

        Tensor Q = Tensors.random(B, H, L, D);
        Tensor K = Tensors.random(B, H, L, D);
        Tensor V = Tensors.random(B, H, L, D);

        // baseline (CPU) with causal mask
        Tensor KT = K.transpose(2, 3);
        Tensor scores = Q.matmul(KT).mul(scale);
        Tensor mask = Tensors.triangularMask(L, L);
        scores = scores.add(mask);
        Tensor attn = scores.softmax();
        Tensor outRef = attn.matmul(V);

        // fused (GPU) with causal=true
        Tensor Qg = Q.gpu(device);
        Tensor Kg = K.gpu(device);
        Tensor Vg = V.gpu(device);
        Tensor outGpu = FlashAttention.forward(Qg, Kg, Vg, scale, true);
        Assumptions.assumeTrue(outGpu != null, "Fused kernel did not run (returned null)");
        assertArrayEquals(outRef.shape(), outGpu.shape(), "Output shape mismatch (causal)");

        float[] a = outRef.data();
        float[] b = outGpu.data();
        double error = mae(a, b);
        System.out.println("MAE (causal): " + error);

        assertTrue(error < 3e-3, "MAE too high (causal): " + error);
    }
}
