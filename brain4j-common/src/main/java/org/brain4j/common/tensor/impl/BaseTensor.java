package org.brain4j.common.tensor.impl;

import org.brain4j.common.Tensors;
import org.brain4j.common.activation.Activation;
import org.brain4j.common.lang.DoubleToDoubleFunction;
import org.brain4j.common.tensor.Tensor;
import org.brain4j.common.tensor.autograd.AutogradContext;
import org.brain4j.common.tensor.autograd.Operation;
import org.brain4j.common.tensor.autograd.impl.*;
import org.brain4j.common.tensor.broadcast.TensorBroadcast;
import org.brain4j.common.tensor.index.Range;
import org.brain4j.common.tensor.parallel.ParallelMap;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.function.Supplier;

import static org.brain4j.common.Tensors.ones;

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
        int start = 0;
        int end = shape[dim];
        int step = 1;

        if (range != null) {
            start = range.start(shape[dim]);
            end = range.end(shape[dim]);
            step = range.step();
        }

        for (int i = start, j = 0; i < end; i += step, j++) {
            srcIndices[dim] = i;
            dstIndices[dim] = j;
            sliceCopy(result, ranges, srcIndices, dstIndices, dim + 1);
        }
    }

    protected void softmax1D(double temperature, float[] data) {
        double max = Double.NEGATIVE_INFINITY;

        for (float value : data) {
            max = Math.max(max, value);
        }

        double sum = 0.0;

        for (float value : data) {
            sum += Math.exp((value - max) / temperature);
        }

        for (int i = 0; i < data.length; i++) {
            double value = Math.exp((data[i] - max) / temperature) / sum;
            data[i] = (float) value;
        }
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
    public int[] shape() {
        return shape;
    }

    @Override
    public float[] data() {
        return data;
    }

    @Override
    public int[] strides() {
        return strides;
    }

    @Override
    public int getLinearIndex(int... indices) {
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
        return data()[getLinearIndex(indices)];
    }

    @Override
    public Tensor set(float value, int... indices) {
        data[getLinearIndex(indices)] = value;
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
    public Tensor add(Tensor other) {
        return TensorBroadcast.add(this, other);
    }

    @Override
    public Tensor add(double value) {
        for (int i = 0; i < data.length; i++) {
            data[i] += (float) value;
        }

        return this;
    }

    @Override
    public Tensor sub(Tensor other) {
        return TensorBroadcast.sub(this, other);
    }

    @Override
    public Tensor sub(double value) {
        for (int i = 0; i < data.length; i++) {
            data[i] -= (float) value;
        }

        return this;
    }

    @Override
    public Tensor mul(Tensor other) {
        return TensorBroadcast.mul(this, other);
    }

    @Override
    public Tensor mul(double value) {
        for (int i = 0; i < data.length; i++) {
            data[i] *= (float) value;
        }

        return this;
    }

    @Override
    public Tensor div(Tensor other) {
        return TensorBroadcast.div(this, other);
    }

    @Override
    public Tensor div(double value) {
        for (int i = 0; i < data.length; i++) {
            data[i] /= (float) value;
        }

        return this;
    }

    @Override
    public Tensor pow(Tensor other) {
        return TensorBroadcast.pow(this, other);
    }

    @Override
    public Tensor pow(double value) {
        for (int i = 0; i < data.length; i++) {
            data[i] = (float) Math.pow(data[i], value);
        }

        return this;
    }

    @Override
    public Tensor sqrt() {
        for (int i = 0; i < data.length; i++) {
            data[i] = (float) Math.sqrt(data[i]);
        }

        return this;
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
        
        return reshape(newShape);
    }

    @Override
    public Tensor squeeze(int dimension) {
        if (dimension >= rank()) {
            throw new IllegalArgumentException("Dimension must be less than the rank!");
        }

        if (shape[dimension] != 1) {
            return this;
        }

        int[] newShape = new int[shape.length - 1];
        int idx = 0;

        for (int i = 0; i < shape.length; i++) {
            if (i != dimension) {
                newShape[idx++] = shape[i];
            }
        }

        return reshape(newShape);
    }

    @Override
    public Tensor unsqueeze() {
        return unsqueeze(0);
    }

    public Tensor unsqueeze(int dim) {
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

        return Tensors.create(newShape, data());
    }

    @Override
    public Tensor transpose() {
        int rank = shape.length;

        if (rank == 1) {
            return reshape(1, elements());
        }

        int[] newShape = shape.clone();
        newShape[rank - 2] = shape[rank - 1];
        newShape[rank - 1] = shape[rank - 2];

        int[] newStrides = strides.clone();
        newStrides[rank - 1] = strides[rank - 2];
        newStrides[rank - 2] = strides[rank - 1];

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
        if (dim < 0 || dim >= shape.length) {
            throw new IllegalArgumentException("Dimension " + dim + " out of bounds for tensor of shape " + Arrays.toString(shape));
        }

        int[] newShape = Tensors.computeNewShape(shape, dim, keepDim);
        int reducedSize = shape[dim];

        Tensor result = Tensors.zeros(newShape);
        float[] resultData = result.data();

        int outerSize = 1;

        for (int i = 0; i < dim; i++) {
            outerSize *= shape[i];
        }

        int innerSize = 1;

        for (int i = dim + 1; i < shape.length; i++) {
            innerSize *= shape[i];
        }

        for (int outer = 0; outer < outerSize; outer++) {
            for (int inner = 0; inner < innerSize; inner++) {
                float sum = 0;

                for (int i = 0; i < reducedSize; i++) {
                    int index = outer * reducedSize * innerSize + i * innerSize + inner;
                    sum += data[index];
                }

                int resultIndex = outer * innerSize + inner;
                resultData[resultIndex] = sum;
            }
        }

        return result;
    }

    @Override
    public Tensor mean(int dim, boolean keepDim) {
        Tensor summed = this.sum(dim, keepDim);

        float divisor = shape[dim];
        float[] resultData = summed.data().clone();

        for (int i = 0; i < resultData.length; i++) {
            resultData[i] /= divisor;
        }

        return Tensors.create(summed.shape(), resultData);
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
            if (i < ranges.length && ranges[i] != null) {
                newShape[i] = ranges[i].size(shape[i]);
            } else {
                newShape[i] = shape[i];
            }
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
        
        return forward(new TransposeOperation());
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
        Tensor result = clone();
        int[] dims = shape;

        int rank = dims.length;
        int lastDim = dims[rank - 1];
        int total = 1;

        for (int i = 0; i < rank - 1; i++) {
            total *= dims[i];
        }

        for (int i = 0; i < total; i++) {
            int[] indices = Tensors.unravelIndex(i, Arrays.copyOf(dims, rank - 1));
            float[] vector = new float[lastDim];

            for (int j = 0; j < lastDim; j++) {
                int[] fullIndex = Arrays.copyOf(indices, rank);
                fullIndex[rank - 1] = j;
                vector[j] = get(fullIndex);
            }

            softmax1D(temperature, vector);

            for (int j = 0; j < lastDim; j++) {
                int[] fullIndex = Arrays.copyOf(indices, rank);
                fullIndex[rank - 1] = j;
                result.set(vector[j], fullIndex);
            }
        }

        return result;
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
        return toString("%.16f");
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
