import org.brain4j.core.Brain4J;
import org.brain4j.math.Tensors;
import org.brain4j.math.gpu.device.Device;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.index.Range;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

public class TestOperations {

    private final Device device;

    public TestOperations() {
        this.device = Brain4J.firstDevice();
        Brain4J.initKernels(device);
    }

    @Test
    public void orthogonalTest() {
        Tensor A = Tensors.orthogonal(3, 3);
        Tensor T = Tensors.matrix(3, 3,
            1, 0, 0,
            0, 1, 0,
            0, 0, 1
        );

        Tensor R = A.transpose().matmul(A);
        assertArrayEquals(R.data(), T.data(), 0.0001f);
    }
    
    @Test
    public void normTest() {
        double epsilon = 1e-5;
        
        Tensor A = Tensors.random(32, 16, 64);
        Tensor B = A.layerNorm(epsilon);
        
        Tensor gpuA = A.to(device);
        Tensor gpuB = gpuA.layerNorm(epsilon);
        
        assertArrayEquals(B.data(), gpuB.data(), 0.001f);
    }

    @Test
    public void convTest() {
        Tensor A = Tensors.random(16, 3, 64, 64);
        Tensor B = Tensors.random(3, 7, 7);
        
        Tensor C = A.convolve(B);
        System.out.println("conv shape = " + Arrays.toString(C.shape()));
    }
    
    @Test
    public void concatTest() {
        Tensor A = Tensors.random(2, 3);
        Tensor B = Tensors.random(2, 2);
        Tensor C = A.concat(B);

        Tensor gpuA = A.gpu(device);
        Tensor gpuB = B.gpu(device);
        Tensor gpuC = gpuA.concat(gpuB);

        assertArrayEquals(C.shape(), gpuC.shape());
        assertArrayEquals(C.data(), gpuC.data(), 0.001f);
    }

    @Test
    public void matmulTest() {
        Tensor A = Tensors.random(4, 8);
        Tensor B = Tensors.random(8, 3);
        Tensor C = A.matmul(B);

        Tensor gpuA = A.gpu(device);
        Tensor gpuB = B.gpu(device);
        Tensor gpuC = gpuA.matmul(gpuB);
        
        assertArrayEquals(C.shape(), gpuC.shape());
        assertArrayEquals(C.data(), gpuC.data(), 0.001f);
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
        assertArrayEquals(B.data(), gpuB.data(), 0.001f);
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

        assertArrayEquals(C.data(), gpuC.data(), 0.001f);
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

        assertArrayEquals(C.data(), gpuC.data(), 0.001f);
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

        assertArrayEquals(C.data(), gpuC.data(), 0.001f);
    }

    @Test
    public void divTest() {
        Tensor A = Tensors.random(32, 32);
        Tensor B = Tensors.random(32, 32);
        Tensor C = A.divide(B);

        Tensor gpuA = A.gpu(device);
        Tensor gpuB = B.gpu(device);
        Tensor gpuC = gpuA.divide(gpuB);

        assertArrayEquals(C.data(), gpuC.data(), 0.001f);
    }

    @Test
    public void chainTest() {
        Tensor A = Tensors.random(16, 16);
        Tensor B = A.matmul(A.transpose()).relu().sum(0, false);

        Tensor gpuA = A.gpu(device);
        Tensor gpuB = gpuA.matmul(gpuA.transpose()).relu().sum(0, false);

        assertEquals(B.get(0), gpuB.get(0), 0.001f);
    }

    @Test
    public void transposeTest() {
        Tensor A = Tensors.random(4, 8);
        Tensor B = Tensors.random(3, 8);
        Tensor C = A.matmul(B.transpose());

        Tensor gpuA = A.gpu(device);
        Tensor gpuB = B.gpu(device);
        Tensor gpuC = gpuA.matmul(gpuB.transpose());

        assertArrayEquals(C.shape(), gpuC.shape());
        assertArrayEquals(C.data(), gpuC.data(), 0.001f);
    }
    
    @Test
    void testTranspose3D() {
        Tensor t = Tensors.create(new int[]{2, 2, 2}, 1, 2,
            3, 4,
            5, 6,
            7, 8);
        
        Tensor transposed = t.transpose(0, 1);
        
        assertArrayEquals(new int[]{2, 2, 2}, transposed.shape());
        
        assertEquals(1, transposed.get(0, 0, 0));
        assertEquals(5, transposed.get(0, 1, 0));
        assertEquals(3, transposed.get(1, 0, 0));
        assertEquals(7, transposed.get(1, 1, 0));
    }
}
