package org.brain4j.core.layer.impl.utility;

import org.brain4j.common.tensor.Tensor;
import org.brain4j.core.importing.proto.ProtoModel;
import org.brain4j.core.layer.Layer;
import org.brain4j.core.training.StatesCache;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

public class ReshapeLayer extends Layer {

    private int[] shape;
    
    public ReshapeLayer() {
    }
    
    public ReshapeLayer(int... shape) {
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
                .addAllShape(Arrays.stream(shape).boxed().collect(Collectors.toList()));
        return List.of(tensorBuilder);
    }
    
    @Override
    public Tensor forward(StatesCache cache, Tensor input, boolean training) {
        int[] inputShape = input.shape();
        int[] newShape = new int[shape.length + 1];

        newShape[0] = inputShape[0];

        System.arraycopy(shape, 0, newShape, 1, shape.length);

        return input.reshapeGrad(newShape);
    }

    @Override
    public int size() {
        return shape[shape.length - 1];
    }
}
