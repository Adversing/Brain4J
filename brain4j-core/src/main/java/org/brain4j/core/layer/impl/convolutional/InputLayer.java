package org.brain4j.core.layer.impl.convolutional;

import org.brain4j.core.importing.proto.ProtoModel;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.core.layer.Layer;
import org.brain4j.core.importing.proto.SerializeUtils;
import org.brain4j.core.training.StatesCache;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

public class InputLayer extends Layer {

    private int[] shape;
    
    public InputLayer() {
    }

    public InputLayer(int... shape) {
        this.shape = shape;
    }

    @Override
    public void deserialize(List<ProtoModel.Tensor> tensors, ProtoModel.Layer layer) {
        for (ProtoModel.Tensor tensor : tensors) {
            if (tensor.getName().equals("shape")) {
                this.shape = tensor.getShapeList().stream().mapToInt(Integer::intValue).toArray();
            }
        }
    }

    @Override
    public List<ProtoModel.Tensor.Builder> weightsList() {
        ProtoModel.Tensor.Builder tensorBuilder =
            ProtoModel.Tensor.newBuilder()
                .setName("shape")
                .addAllShape(Arrays.stream(shape)
                    .boxed()
                    .collect(Collectors.toList()));
        return List.of(tensorBuilder);
    }

    @Override
    public Tensor[] forward(StatesCache cache, Tensor... inputs) {
        for (Tensor input : inputs) {
            int[] inputShape = input.shape();

            if (input.rank() > shape.length + 1) {
                throw new IllegalArgumentException(
                    String.format("Input rank mismatch! Expected input with shape %s but got %s",
                        Arrays.toString(shape), Arrays.toString(inputShape)
                    )
                );
            }
        }

        return inputs;
    }

    @Override
    public int size() {
        return shape[shape.length - 1];
    }

    @Override
    public boolean validInput(Tensor input) {
        if (input.rank() > shape.length + 1) return false;

        int[] inputShape = input.shape();
        // Leniency, input COULD have a batch, it's not guaranteed
        int offset = inputShape.length - shape.length;

        for (int i = 0; i < shape.length; i++) {
            if (inputShape[i + offset] != shape[i]) return false;
        }

        return true;
    }
}
