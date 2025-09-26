package org.brain4j.math.tensor.impl;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;
import org.brain4j.math.Tensors;
import org.brain4j.math.activation.Activation;
import org.brain4j.math.gpu.device.DeviceUtils;
import org.brain4j.math.lang.DoubleToDoubleFunction;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.autograd.AutogradContext;
import org.brain4j.math.tensor.autograd.Operation;
import org.brain4j.math.tensor.autograd.impl.*;
import org.brain4j.math.tensor.index.Range;
import org.brain4j.math.tensor.parallel.ParallelMap;
import org.brain4j.math.tensor.sum.TensorReducer;
import org.brain4j.math.tensor.sum.impl.ScalarTensorReducer;
import org.brain4j.math.tensor.sum.impl.SimdTensorReducer;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.function.Supplier;
import java.util.stream.IntStream;

import static org.brain4j.math.Tensors.ones;
import static org.brain4j.math.Tensors.unravelIndex;

public abstract class BaseTensor implements Tensor, Cloneable {

    protected AutogradContext autogradContext;
    protected int[] shape;
    protected int[] strides;
    protected float[] data;
    protected boolean transposed;

    protected void appendTensor(StringBuilder result, int dim, int[] indices, String format) {
        if (dim == shape.length - 1) {
            result.append("[");
            for (int i = 0; i < shape[dim]; i++) {
                indices[dim] = i;

                float value = get(indices);
                result.append(format.formatted(value));

                if (i < shape[dim] - 1) {
                    result.append(", ");
                }
            }
            result.append("]");
        } else {
            result.append("[");
            for (int i = 0; i < shape[dim]; i++) {
                indices[dim] = i;
                appendTensor(result, dim + 1, indices, format);

                if (i < shape[dim] - 1) {
                    result.append(",\n");
                    result.append(" ".repeat(dim + 1));
                }
            }

            result.append("]");
        }
    }

    protected void sliceCopy(
        Tensor result,
        Range[] ranges,
        int[] srcIndices,
        int[] dstIndices,
        int dim
    ) {
        if (dim == shape.length) {
            result.set(get(srcIndices), dstIndices);
            return;
        }

        Range range = dim < ranges.length ? ranges[dim] : null;
        int dimension = shape[dim];

        int start = range == null ? 0 : range.start(dimension);
        int end = range == null ? shape[dim] : range.end(dimension);
        int step = range == null ? 1 : range.step();

        if (dim == shape.length - 1 && step == 1) {
            int length = end - start;

            srcIndices[dim] = start;
            dstIndices[dim] = 0;

            int srcOffset = Tensors.flattenIndex(srcIndices, this.strides);
            int dstOffset = Tensors.flattenIndex(dstIndices, result.strides());

            System.arraycopy(
                this.data, srcOffset,
                result.data(), dstOffset,
                length
            );
            return;
        }

        for (int i = start, j = 0; i < end; i += step, j++) {
            srcIndices[dim] = i;
            dstIndices[dim] = j;
            sliceCopy(result, ranges, srcIndices, dstIndices, dim + 1);
        }
    }
    
    protected void softmax1D(float[] data, int offset, int length, double temperature) {
        float max = Float.NEGATIVE_INFINITY;
        
        for (int i = 0; i < length; i++) {
            float value = data[offset + i];

            if (value > max) {
                max = value;
            }
        }

        float sum = 0f;
        
        for (int i = 0; i < length; i++) {
            float e = (float) Math.exp((data[offset + i] - max) / temperature);
            sum += e;
            data[offset + i] = e;
        }
        
        for (int i = 0; i < length; i++) {
            data[offset + i] /= sum;
        }
    }
    
    protected Tensor softmax1D(double temperature) {
        Tensor result = clone();
        softmax1D(result.data(), 0, result.elements(), temperature);
        return result;
    }
    
    protected Tensor softmax2D(double temperature) {
        Tensor result = clone();
        
        int rows = shape[0];
        int cols = shape[1];
        
        for (int r = 0; r < rows; r++) {
            softmax1D(result.data(), r * cols, cols, temperature);
        }
        
        return result;
    }
    
    protected Tensor softmax3D(double temperature) {
        Tensor result = clone();
        
        int batches = shape[0];
        int rows = shape[1];
        int cols = shape[2];
        
        float[] data = result.data();
        int strideBR = rows * cols;
        
        IntStream.range(0, batches * rows).parallel().forEach(i -> {
            int b = i / rows;
            int r = i % rows;
            softmax1D(data, b * strideBR + r * cols, cols, temperature);
        });
        
        return result;
    }
    
    protected Tensor softmaxND(double temperature) {
        Tensor result = clone();
        
        int rank = shape.length;
        int lastDim = shape[rank - 1];
        int outerSize = 1;
        
        for (int i = 0; i < rank - 1; i++) {
            outerSize *= shape[i];
        }
        
        float[] data = result.data();

        IntStream.range(0, outerSize)
            .parallel()
            .forEach(outer -> softmax1D(data, outer * lastDim, lastDim, temperature));

        return result;
    }
    
    protected void setSliceRecursive(int[] index, int dim, int offset, int sliceSize, Tensor input) {
        if (dim == index.length - 1) {
            for (int i = 0; i < sliceSize; i++) {
                index[dim] = offset + i;

                int[] inputIndex = Arrays.copyOf(index, index.length);
                inputIndex[dim] = i;

                float value = input.get(inputIndex);
                this.set(value, index);
            }
            return;
        }

        for (int i = 0; i < this.shape()[dim]; i++) {
            index[dim] = i;
            setSliceRecursive(index, dim + 1, offset, sliceSize, input);
        }
    }

    @Override
    public int shape(int index) {
        return shape[index];
    }

    @Override
    public int[] shape() {
        return shape;
    }

    @Override
    public float[] data() {
        return data;
    }

    @Override
    public float[] toArray() {
        if (!transposed) return data;

        float[] result = new float[data.length];

        for (int i = 0; i < data.length; i++) {
            int[] srcIndices = unravelIndex(i, shape);
            result[i] = data[linearIndex(srcIndices)];
        }

        return result;
    }

    @Override
    public int[] strides() {
        return strides;
    }
    
    @Override
    public byte[] toByteArray() {
        float[] data = data(); // gpu workaround
        
        ByteBuffer buffer = ByteBuffer.allocate(data.length * 4);
        buffer.order(ByteOrder.LITTLE_ENDIAN);
        
        for (float value : data) {
            buffer.putFloat(value);
        }
        
        return buffer.array();
    }
    
    @Override
    public int linearIndex(int... indices) {
        if (indices.length != shape.length) {
            throw new IllegalArgumentException(
                "The shape of the tensor does not match the number of indices");
        }

        for (int i = 0; i < indices.length; i++) {
            if (indices[i] < 0 || indices[i] >= shape[i]) {
                throw new IndexOutOfBoundsException(
                    "Index " + indices[i] + " for dimension " + i + " is out of bounds [0, " + shape[i] + ")"
                );
            }
        }

        return Tensors.flattenIndex(indices, strides);
    }

    @Override
    public float get(int... indices) {
        return data()[linearIndex(indices)];
    }

    @Override
    public Tensor set(float value, int... indices) {
        data[linearIndex(indices)] = value;
        return this;
    }

    @Override
    public int rank() {
        return shape.length;
    }

    @Override
    public int elements() {
        return data().length;
    }

    @Override
    public int argmax() {
        float[] data = data();

        double max = Double.NEGATIVE_INFINITY;
        int maxIndex = 0;

        for (int i = 0; i < data.length; i++) {
            if (data[i] > max) {
                max = data[i];
                maxIndex = i;
            }
        }

        return maxIndex;
    }

    @Override
    public Tensor clone() {
        try {
            BaseTensor copy = (BaseTensor) super.clone();

            copy.shape = shape.clone();
            copy.strides = strides.clone();
            copy.data = data.clone();
            copy.autogradContext = null;

            return copy;
        } catch (CloneNotSupportedException e) {
            throw new RuntimeException(e);
        }
    }
    
    @Override
    public Tensor flatten() {
        return reshape(elements());
    }

    @Override
    public Tensor convolve(Tensor kernel) {
        return Tensors.convolve(this, kernel);
    }

    @Override
    public Tensor layerNorm(double epsilon) {
        int rank = shape.length;
        int featuresSize = shape[rank - 1];
        
        int batchSize = 1;
        
        for (int i = 0; i < rank - 1; i++) batchSize *= shape[i];
        
        for (int batchIdx = 0; batchIdx < batchSize; batchIdx++) {
            int base = 0;
            int rem = batchIdx;
            
            for (int dim = rank - 2; dim >= 0; dim--) {
                int idxDim = rem % shape[dim];
                rem /= shape[dim];
                base += idxDim * strides[dim];
            }
            
            float mean = 0f;
            
            for (int j = 0; j < featuresSize; j++) {
                mean += data[base + j * strides[rank - 1]];
            }
            
            mean /= featuresSize;
            
            float var = 0f;
            
            for (int j = 0; j < featuresSize; j++) {
                float x = data[base + j * strides[rank - 1]];
                float diff = x - mean;
                
                var += diff * diff;
            }
            
            var /= featuresSize;
            
            float denom = (float)Math.sqrt(var + epsilon);
            
            for (int j = 0; j < featuresSize; j++) {
                int idx = base + j * strides[rank - 1];
                data[idx] = (data[idx] - mean) / denom;
            }
        }

        return this;
    }

    @Override
    public double distance(Tensor other) {
        return Math.sqrt(distanceSquared(other));
    }

    @Override
    public double distanceSquared(Tensor other) {
        double sum = 0;
        float[] cached = data();

        for (int i = 0; i < cached.length; i++) {
            double diff = cached[i] - other.data()[i];
            sum += diff * diff;
        }

        return sum;
    }
    
    @Override
    public Tensor squeeze() {
        int count = 0;
        
        for (int dim : shape) {
            if (dim != 1) count++;
        }
        
        if (count == rank()) {
            return this;
        }
        
        int[] newShape = new int[count];
        int idx = 0;
        
        for (int dim : shape) {
            if (dim != 1) {
                newShape[idx++] = dim;
            }
        }
        
        Tensor reshaped = reshape(newShape);
        reshaped.setAutogradContext(autogradContext);
        
        return reshaped;
    }

    @Override
    public Tensor squeeze(int dim) {
        dim %= shape.length;

        if (dim >= rank()) {
            throw new IllegalArgumentException("Dimension must be less than the rank!");
        }

        if (shape[dim] != 1) {
            return this;
        }

        int[] newShape = new int[shape.length - 1];
        int idx = 0;

        for (int i = 0; i < shape.length; i++) {
            if (i != dim) {
                newShape[idx++] = shape[i];
            }
        }

        Tensor result = reshape(newShape);
        result.setAutogradContext(autogradContext);
        
        return result;
    }

    @Override
    public Tensor unsqueeze() {
        return unsqueeze(0);
    }
    
    @Override
    public Tensor broadcastLike(Tensor other) {
        int[] targetShape = other.shape();
        
        if (Arrays.equals(shape, targetShape)) {
            return this;
        }
        
        int targetRank = targetShape.length;
        int srcRank = shape.length;
        
        int[] alignedSrcShape = new int[targetRank];
        int pad = targetRank - srcRank;
        
        for (int i = 0; i < pad; i++) alignedSrcShape[i] = 1;
        System.arraycopy(shape, 0, alignedSrcShape, pad, srcRank);
        
        for (int d = 0; d < targetRank; d++) {
            if (alignedSrcShape[d] > targetShape[d]) {
                throw new IllegalArgumentException("Cannot broadcast: source dimension " +
                    alignedSrcShape[d] + " > target dimension " + targetShape[d] + " at axis " + d);
            }
        }
        
        Tensor out = Tensors.zeros(targetShape);
        
        int total = out.elements();
        int[] srcCoords = new int[targetRank];
        
        for (int i = 0; i < total; i++) {
            int[] coords = Tensors.unravelIndex(i, targetShape);
            boolean useZero = false;
            
            for (int d = 0; d < targetRank; d++) {
                int s = alignedSrcShape[d];
                if (s == targetShape[d]) {
                    srcCoords[d] = coords[d];
                } else if (s == 1) {
                    srcCoords[d] = 0;
                } else {
                    if (coords[d] < s) {
                        srcCoords[d] = coords[d];
                    } else {
                        useZero = true;
                        break;
                    }
                }
            }
            
            if (!useZero) {
                out.set(get(srcCoords), coords);
            }
        }
        
        return out;
    }
    
    public Tensor unsqueeze(int dim) {
        dim %= shape.length;

        if (dim < 0 || dim > shape.length) {
            throw new IllegalArgumentException("Invalid dimension for unsqueeze: " + dim);
        }

        int[] newShape = new int[shape.length + 1];

        for (int i = 0, j = 0; i < newShape.length; i++) {
            if (i == dim) {
                newShape[i] = 1;
            } else {
                newShape[i] = shape[j++];
            }
        }
        
        Tensor result = reshape(newShape);
        result.setAutogradContext(autogradContext);
        
        return result;
    }

    @Override
    public Tensor transpose() {
        int rank = shape.length;
        return transpose(rank - 2, rank - 1);
    }
    
    @Override
    public Tensor transpose(int dim1, int dim2) {
        int rank = shape.length;
        
        if (rank == 1) {
            return reshape(1, elements());
        }
        
        int[] newShape = shape.clone();
        newShape[dim1] = shape[dim2];
        newShape[dim2] = shape[dim1];
        
        int[] newStrides = strides.clone();
        newStrides[dim2] = strides[dim1];
        newStrides[dim1] = strides[dim2];
        
        BaseTensor view = (BaseTensor) Tensors.create(newShape, newStrides, data);
        view.transposed = !transposed;
        
        return view;
    }
    
    @Override
    public boolean transposed() {
        return transposed;
    }

    @Override
    public double sum() {
        double sum = 0.0;
        float[] data = data();

        for (float value : data) {
            sum += value;
        }

        return sum;
    }

    @Override
    public double mean() {
        return sum() / elements();
    }

    @Override
    public double variance() {
        double mean = mean();
        double variance = 0.0;

        for (float value : data) {
            variance += Math.pow(value - mean, 2);
        }

        return variance / data.length;
    }

    @Override
    public double max() {
        double max = Double.NEGATIVE_INFINITY;

        for (float value : data) {
            max = Math.max(max, value);
        }

        return max;
    }

    @Override
    public double min() {
        double min = Double.POSITIVE_INFINITY;

        for (float value : data) {
            min = Math.min(min, value);
        }

        return min;
    }

    @Override
    public Tensor sum(int dim, boolean keepDim) {
        dim %= shape.length;

        TensorReducer reducer = DeviceUtils.isSimdAvailable() ? new SimdTensorReducer() : new ScalarTensorReducer();
        Tensor result = reducer.sum(this, dim, keepDim);
        
        result.setAutogradContext(autogradContext);
        return result;
    }

    @Override
    public Tensor mean(int dim, boolean keepDim) {
        dim %= shape.length;

        Tensor summed = this.sum(dim, keepDim);

        float divisor = shape[dim];
        float[] resultData = summed.data().clone();

        for (int i = 0; i < resultData.length; i++) {
            resultData[i] /= divisor;
        }

        Tensor result = Tensors.create(summed.shape(), resultData);
        result.setAutogradContext(autogradContext);
        
        return result;
    }
    
    @Override
    public Tensor variance(int dim, boolean keepDim) {
        return variance(mean(dim, keepDim), dim, keepDim);
    }
    
    @Override
    public Tensor variance(Tensor mean, int dim, boolean keepDim) {
        dim %= shape.length;

        Tensor meanFirstSquare = clone().pow(2).mean(dim, keepDim);
        Tensor meanSecondSquare = mean.clone().pow(2);
        
        return meanFirstSquare.sub(meanSecondSquare);
    }
    
    @Override
    public Tensor sign() {
        Tensor result = Tensors.zeros(shape);

        for (int i = 0; i < data.length; i++) {
            result.data()[i] = Math.signum(data[i]);
        }

        return result;
    }

    @Override
    public Tensor reshape(int... newShape) {
        int newSize = Tensors.computeSize(newShape);

        if (newSize != data().length) {
            throw new IllegalArgumentException(
                    "The total new dimension (" + newSize + ") does not match the current dimension (" + data().length + ")"
            );
        }

        return Tensors.create(newShape, data());
    }

    @Override
    public Tensor concat(Tensor other) {
        if (shape.length != other.shape().length) {
            throw new IllegalArgumentException("Concatenation is only supported for tensors with the same number of dimensions.");
        }

        for (int i = 0; i < shape.length - 1; i++) {
            if (shape[i] != other.shape()[i]) {
                throw new IllegalArgumentException("Shapes must match on all dimensions except the last.");
            }
        }

        int rank = shape.length;
        int lastDim = shape[rank - 1];
        int otherLastDim = other.shape()[rank - 1];
        int concatLastDim = lastDim + otherLastDim;

        int[] newShape = Arrays.copyOf(shape, rank);
        newShape[rank - 1] = concatLastDim;

        int outerSize = 1;
        for (int i = 0; i < rank - 1; i++) {
            outerSize *= shape[i];
        }

        float[] resultData = new float[outerSize * concatLastDim];
        float[] thisData = this.data();
        float[] otherData = other.data();

        for (int i = 0; i < outerSize; i++) {
            System.arraycopy(thisData, i * lastDim, resultData, i * concatLastDim, lastDim);
            System.arraycopy(otherData, i * otherLastDim, resultData, i * concatLastDim + lastDim, otherLastDim);
        }

        return Tensors.create(newShape, resultData);
    }

    @Override
    public Tensor concat(Tensor other, int dimension) {
        if (shape.length != other.shape().length) {
            throw new IllegalArgumentException("Tensors must have the same rank.");
        }

        int rank = rank();

        if (dimension < 0 || dimension >= rank) {
            throw new IllegalArgumentException("Invalid dimension: " + dimension);
        }

        for (int i = 0; i < rank; i++) {
            if (i != dimension && shape[i] != other.shape()[i]) {
                throw new IllegalArgumentException("Shapes must match in all dimensions except the concatenation one.");
            }
        }

        int[] newShape = Arrays.copyOf(shape, rank);
        newShape[dimension] += other.shape()[dimension];

        int blockSize = 1;
        int numBlocks = 1;

        for (int i = dimension + 1; i < rank; i++) blockSize *= shape[i];
        for (int i = 0; i < dimension; i++) numBlocks *= shape[i];

        int thisDim = shape[dimension];
        int otherDim = other.shape()[dimension];
        float[] result = new float[numBlocks * (thisDim + otherDim) * blockSize];

        float[] a = this.data();
        float[] b = other.data();

        int resultOffset = 0;
        int thisOffset = 0;
        int otherOffset = 0;

        for (int i = 0; i < numBlocks; i++) {
            int thisBlock = thisDim * blockSize;
            int otherBlock = otherDim * blockSize;

            System.arraycopy(a, thisOffset, result, resultOffset, thisBlock);
            thisOffset += thisBlock;
            resultOffset += thisBlock;

            System.arraycopy(b, otherOffset, result, resultOffset, otherBlock);
            otherOffset += otherBlock;
            resultOffset += otherBlock;
        }

        return Tensors.create(newShape, result);
    }

    @Override
    public Tensor activate(Activation activation) {
        return activation.activate(this);
    }

    @Override
    public Tensor select(int dim, int index) {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    @Override
    public Tensor slice(Range... ranges) {
        if (ranges.length > shape.length) {
            throw new IllegalArgumentException("Too many ranges specified");
        }

        int[] newShape = new int[shape.length];

        for (int i = 0; i < shape.length; i++) {
            int dim = shape[i];
            Range range = i >= ranges.length ? null : ranges[i];
            newShape[i] = range != null ? range.size(dim) : dim;
        }

        Tensor result = new CpuTensor(newShape);

        int[] srcIndices = new int[shape.length];
        int[] dstIndices = new int[shape.length];

        sliceCopy(result, ranges, srcIndices, dstIndices, 0);

        return result;
    }

    @Override
    public void setSliceAlongLastDim(int offset, Tensor input) {
        int[] inputShape = input.shape();
        int[] thisShape = this.shape();

        int rank = thisShape.length;
        int sliceSize = inputShape[rank - 1];

        int[] current = new int[rank];
        setSliceRecursive(current, 0, offset, sliceSize, input);
    }

    @Override
    public Tensor mask(float[] mask) {
        if (mask.length != data.length) {
            throw new IllegalArgumentException("Mask length must be as long as the data");
        }

        for (int i = 0; i < mask.length; i++) {
            data[i] = Math.max(0, data[i] - mask[i]);
        }

        return this;
    }

    @Override
    public Tensor map(DoubleToDoubleFunction function) {
        ParallelMap.map(function, data);
        return this;
    }

    @Override
    public Tensor fill(float value) {
        Arrays.fill(data, value);
        return this;
    }

    @Override
    public Tensor fill(Supplier<Double> supplier) {
        for (int i = 0; i < data.length; i++) {
            data[i] = supplier.get().floatValue();
        }

        return this;
    }

    @Override
    public AutogradContext autogradContext() {
        return autogradContext;
    }

    @Override
    public void setAutogradContext(AutogradContext context) {
        this.autogradContext = context;
    }

    @Override
    public Tensor withGrad() {
        this.autogradContext = new AutogradContext(true);
        return this;
    }

    @Override
    public boolean usesGrad() {
        return autogradContext != null && autogradContext.requiresGrad();
    }

    @Override
    public void zerograd() {
        if (autogradContext != null) {
            autogradContext.zerograd();
        }
    }

    @Override
    public Tensor grad() {
        if (autogradContext != null) {
            return autogradContext.getGrad();
        }

        return null;
    }

    @Override
    public void backward() {
        backward(ones(shape));
    }

    @Override
    public void backward(Tensor gradOutput) {
        if (autogradContext == null) {
            throw new IllegalArgumentException("Autograd is not enabled for this tensor");
        }

        autogradContext.backward(gradOutput);
    }

    @Override
    public Tensor forward(Operation operation) {
        if (operation.requiredInputs() != 1) {
            throw new IllegalArgumentException(
                "This operation requires " + operation.requiredInputs() + " inputs! Received 1 instead."
            );
        }

        Tensor result = operation.compute(this);

        if (result.autogradContext() == null) {
            result.setAutogradContext(new AutogradContext(true));
        }

        result.autogradContext().setOperation(operation, this);
        return result;
    }

    @Override
    public Tensor forward(Operation operation, Tensor other) {
        if (operation.requiredInputs() != 2) {
            throw new IllegalArgumentException(
                "This operation requires " + operation.requiredInputs() + " inputs! Received 2 instead."
            );
        }

        Tensor result = operation.compute(this, other);

        if (result.autogradContext() == null) {
            result.setAutogradContext(new AutogradContext(true));
        }

        result.autogradContext().setOperation(operation, this, other);
        return result;
    }

    @Override
    public Tensor forward(Operation operation, Tensor... others) {
        List<Tensor> allInputs = new ArrayList<>();

        allInputs.add(this);
        allInputs.addAll(Arrays.asList(others));

        if (allInputs.size() != operation.requiredInputs()) {
            throw new IllegalArgumentException(
                "This operation requires " + operation.requiredInputs() + " inputs! Received " + allInputs.size()
                    + " instead."
            );
        }

        Tensor[] allInputsArray = allInputs.toArray(new Tensor[0]);
        Tensor result = operation.compute(allInputsArray);

        if (result.autogradContext() == null) {
            result.setAutogradContext(new AutogradContext(true));
        }

        result.autogradContext().setOperation(operation, allInputsArray);
        return result;
    }

    @Override
    public Tensor addGrad(Tensor other) {
        if (!usesGrad() && !other.usesGrad()) {
            throw new IllegalArgumentException("At least one of the two tensors should be used with backflow!");
        }

        return forward(new AddOperation(), other);
    }

    @Override
    public Tensor mulGrad(Tensor other) {
        if (!usesGrad() && !other.usesGrad()) {
            throw new IllegalArgumentException("At least one of the two tensors should be used with backflow!");
        }

        return forward(new MulOperation(), other);
    }

    @Override
    public Tensor divGrad(Tensor other) {
        if (!usesGrad() && !other.usesGrad()) {
            throw new IllegalArgumentException("At least one of the two tensors should be used with backflow!");
        }

        return forward(new DivOperation(), other);
    }

    @Override
    public Tensor subGrad(Tensor other) {
        if (!usesGrad() && !other.usesGrad()) {
            throw new IllegalArgumentException("At least one of the two tensors should be used with backflow!");
        }

        return forward(new SubOperation(), other);
    }
    
    @Override
    public Tensor sliceGrad(Range... ranges) {
        if (!usesGrad()) {
            throw new IllegalArgumentException("This tensor does not use backflow!");
        }
        
        return forward(new SliceOperation(ranges));
    }
    
    @Override
    public Tensor matmulGrad(Tensor other) {
        if (!usesGrad() && !other.usesGrad()) {
            throw new IllegalArgumentException("At least one of the two tensors should be used with backflow!");
        }

        return forward(new MatMulOperation(), other);
    }

    @Override
    public Tensor convolveGrad(Tensor other) {
        if (!usesGrad() && !other.usesGrad()) {
            throw new IllegalArgumentException("At least one of the two tensors should be used with backflow!");
        }

        return forward(new ConvolveOperation(), other);
    }

    @Override
    public Tensor transposeGrad() {
        if (!usesGrad()) {
            throw new IllegalArgumentException("This teensors should be used with backflow!");
        }

        int rank = rank();
        return forward(new TransposeOperation(rank - 2, rank - 1));
    }

    @Override
    public Tensor transposeGrad(int dim1, int dim2) {
        if (!usesGrad()) {
            throw new IllegalArgumentException("This teensors should be used with backflow!");
        }

        return forward(new TransposeOperation(dim1, dim2));
    }

    @Override
    public Tensor activateGrad(Activation activation) {
        if (!usesGrad()) {
            throw new IllegalArgumentException("Tensor does not use backflow!");
        }

        return forward(new ActivationOperation(activation));
    }
    
    @Override
    public Tensor concatGrad(Tensor other, int dim) {
        if (!usesGrad() && !other.usesGrad()) {
            throw new IllegalArgumentException("At least one of the two tensors should be used with backflow!");
        }

        return forward(new ConcatOperation(dim), other);
    }

    @Override
    public Tensor reshapeGrad(int... newShape) {
        if (!usesGrad()) {
            throw new IllegalArgumentException("Tensor does not use backflow!");
        }

        return forward(new ReshapeOperation(newShape));
    }

    @Override
    public Tensor squeezeGrad() {
        if (!usesGrad()) {
            throw new IllegalArgumentException("Tensor does not use backflow!");
        }

        return forward(new SqueezeOperation());
    }

    @Override
    public Tensor squeezeGrad(int dimension) {
        if (!usesGrad()) {
            throw new IllegalArgumentException("Tensor does not use backflow!");
        }

        return forward(new SqueezeOperation(dimension));
    }

    @Override
    public Tensor flip() {
        Tensor result = Tensors.zeros(shape);
        int dims = shape.length;

        int total = elements();

        for (int linear = 0; linear < total; linear++) {
            int[] indices = Tensors.unravelIndex(linear, shape);
            int[] flipped = indices.clone();

            if (dims >= 2) {
                flipped[dims - 1] = shape[dims - 1] - 1 - indices[dims - 1];
                flipped[dims - 2] = shape[dims - 2] - 1 - indices[dims - 2];
            }

            float value = this.get(indices);
            result.set(value, flipped);
        }

        return result;
    }

    @Override
    public Tensor softmax() {
        return softmax(1.0);
    }
    
    @Override
    public Tensor softmax(double temperature) {
        return switch (rank()) {
            case 1 -> softmax1D(temperature);
            case 2 -> softmax2D(temperature);
            case 3 -> softmax3D(temperature);
            default -> softmaxND(temperature);
        };
    }
    
    @Override
    public String toString(String format) {
        if (shape.length == 0) {
            return format.formatted(data[0]);
        }

        StringBuilder result = new StringBuilder();
        appendTensor(result, 0, new int[shape.length], format);

        return result.toString();
    }

    @Override
    public String toString() {
        return toString("%.3f");
    }

    @Override
    public Iterator<Float> iterator() {
        return new Iterator<>() {
            private int currentIndex = 0;

            @Override
            public boolean hasNext() {
                return currentIndex < data.length;
            }

            @Override
            public Float next() {
                return data[currentIndex++];
            }
        };
    }
}
