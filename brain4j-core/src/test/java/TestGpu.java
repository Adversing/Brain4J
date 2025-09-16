import org.brain4j.core.Brain4J;
import org.brain4j.math.activation.Activations;
import org.brain4j.math.Tensors;
import org.brain4j.math.gpu.device.Device;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.index.Range;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.*;

public class TestGpu {

    private final Device device;

    public TestGpu() {
        this.device = Brain4J.firstDevice();
        Brain4J.initKernels(device);
    }

    @Test
    public void concatTest() {
        Device device = Brain4J.firstDevice();
        Brain4J.initKernels(device);

        Tensor A = Tensors.random(2, 3);
        Tensor B = Tensors.random(2, 2);
        Tensor C = A.concat(B);

        Tensor gpuA = A.gpu(device);
        Tensor gpuB = B.gpu(device);
        Tensor gpuC = gpuA.concat(gpuB);

        assertArrayEquals(C.shape(), gpuC.shape());
        assertArrayEquals(C.data(), gpuC.data(), 0.0001f);
    }

    @Test
    public void matmulTest() {
        Device device = Brain4J.firstDevice();
        Brain4J.initKernels(device);

        Tensor A = Tensors.random(64, 128);
        Tensor B = Tensors.random(128, 256);
        Tensor C = A.matmul(B);

        Tensor gpuA = A.gpu(device);
        Tensor gpuB = B.gpu(device);
        Tensor gpuC = gpuA.matmul(gpuB);

        assertArrayEquals(C.shape(), gpuC.shape());
        assertArrayEquals(C.data(), gpuC.data(), 0.0001f);
    }

    @Test
    public void sliceTest() {
        Device device = Brain4J.firstDevice();
        Brain4J.initKernels(device);

        Range[] ranges = { Range.all(), Range.interval(10, 20) };

        Tensor A = Tensors.random(64, 32);
        Tensor B = A.slice(ranges);

        Tensor gpuA = A.gpu(device);
        Tensor gpuB = gpuA.slice(ranges);

        assertArrayEquals(B.shape(), gpuB.shape());
        assertArrayEquals(B.data(), gpuB.data(), 0.0001f);
    }

    @Test
    public void addTest() {
        Device device = Brain4J.firstDevice();
        Brain4J.initKernels(device);

        Tensor A = Tensors.random(32, 32);
        Tensor B = Tensors.random(32, 32);
        Tensor C = A.plus(B);

        Tensor gpuA = A.gpu(device);
        Tensor gpuB = B.gpu(device);
        Tensor gpuC = gpuA.plus(gpuB);

        assertArrayEquals(C.data(), gpuC.data(), 0.0001f);
    }

    @Test
    public void subTest() {
        Device device = Brain4J.firstDevice();
        Brain4J.initKernels(device);

        Tensor A = Tensors.random(32, 32);
        Tensor B = Tensors.random(32, 32);
        Tensor C = A.minus(B);

        Tensor gpuA = A.gpu(device);
        Tensor gpuB = B.gpu(device);
        Tensor gpuC = gpuA.minus(gpuB);

        assertArrayEquals(C.data(), gpuC.data(), 0.0001f);
    }

    @Test
    public void mulTest() {
        Device device = Brain4J.firstDevice();
        Brain4J.initKernels(device);

        Tensor A = Tensors.random(32, 32);
        Tensor B = Tensors.random(32, 32);
        Tensor C = A.times(B);

        Tensor gpuA = A.gpu(device);
        Tensor gpuB = B.gpu(device);
        Tensor gpuC = gpuA.times(gpuB);

        assertArrayEquals(C.data(), gpuC.data(), 0.0001f);
    }

    @Test
    public void divTest() {
        Tensor A = Tensors.random(32, 32);
        Tensor B = Tensors.random(32, 32);
        Tensor C = A.divide(B);

        Tensor gpuA = A.gpu(device);
        Tensor gpuB = B.gpu(device);
        Tensor gpuC = gpuA.divide(gpuB);

        assertArrayEquals(C.data(), gpuC.data(), 0.0001f);
    }

    @Test
    public void chainTest() {
        Tensor A = Tensors.random(16, 16);
        Tensor B = A.matmul(A.transpose()).relu().sum(0, false);

        Tensor gpuA = A.gpu(device);
        Tensor gpuB = gpuA.matmul(gpuA.transpose()).relu().sum(0, false);

        assertEquals(B.get(0), gpuB.get(0), 0.0001f);
    }

    @Test
    public void transposeTest() {
        Tensor A = Tensors.random(1, 2, 3, 4);
        Tensor B = A.transpose();

        Tensor gpuA = A.gpu(device);
        Tensor gpuB = gpuA.transpose();

        System.out.println(Arrays.toString(B.shape()));
        System.out.println(Arrays.toString(gpuB.shape()));

        assertArrayEquals(B.shape(), gpuB.shape());
        assertArrayEquals(B.data(), gpuB.data(), 0.0001f);
    }
}
