package org.brain4j.core.layer.impl.transformer;

import org.brain4j.math.Tensors;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.core.importing.proto.ProtoModel;
import org.brain4j.core.importing.proto.SerializeUtils;
import org.brain4j.core.layer.Layer;
import org.brain4j.core.training.StatesCache;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class PosEncodeLayer extends Layer {

    private final Map<Integer, Tensor> preGenerated = new HashMap<>();
    private int dimension;
    private int length = 5000;

    private PosEncodeLayer() {
    }

    public PosEncodeLayer(int length, int dimension) {
        this.length = length;

        for (int i = 0; i < length; i++) {
            preGenerated.put(i, generate(i, dimension));
        }
    }

    @Override
    public Tensor[] forward(StatesCache cache, Tensor... inputs) {
        throwIfTooManyInputs(1, inputs);

        // [batch_size, seq_length, dimension]
        Tensor input = inputs[0];
        int[] shape = input.shape();

        if (shape.length != 3) {
            throw new IllegalArgumentException(
                "Expected input shape [batch_size, seq_length, dimension], got: " + Arrays.toString(shape)
            );
        }

        int seqLength = shape[1];
        int dimension = shape[2];

        Tensor positional = Tensors.zeros(seqLength, dimension);
        float[] posData = positional.data();

        for (int i = 0; i < seqLength; i++) {
            Tensor add = preGenerated.computeIfAbsent(i, index -> generate(index, dimension));
            float[] addData = add.data();

            int index = i * dimension;

            System.arraycopy(addData, 0, posData, index, addData.length);
        }

        return new Tensor[] { input.add(positional) };
    }

    @Override
    public int size() {
        return 0;
    }

    @Override
    public void serialize(ProtoModel.Layer.Builder builder) {
        builder.putAttrs("length", SerializeUtils.value(length));
        builder.putAttrs("dimension", SerializeUtils.value(dimension));
    }

    @Override
    public void deserialize(List<ProtoModel.Tensor> tensors, ProtoModel.Layer layer) {
        this.length = SerializeUtils.attribute(layer, "length", 0);
        this.dimension = SerializeUtils.attribute(layer, "dimension", 0);
    }

    public Tensor generate(int position, int embeddingDim) {
        Tensor token = Tensors.zeros(embeddingDim);

        for (int i = 0; i < embeddingDim; i++) {
            double exponent = (2.0 * Math.floor(i / 2.0)) / embeddingDim;

            double angle = position / Math.pow(10000, exponent);
            double value = (i % 2 == 0) ? Math.sin(angle) : Math.cos(angle);

            token.set(value, i);
        }

        return token.reshape(1, embeddingDim);
    }
    
    @Override
    public List<ProtoModel.Tensor.Builder> weightsList() {
        return List.of();
    }
}
