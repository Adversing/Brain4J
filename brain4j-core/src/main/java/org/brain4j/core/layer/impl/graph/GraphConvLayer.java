package org.brain4j.core.layer.impl.graph;

import com.google.gson.JsonObject;
import org.brain4j.core.activation.Activations;
import org.brain4j.core.layer.Layer;
import org.brain4j.core.training.StatesCache;
import org.brain4j.math.Tensors;
import org.brain4j.math.activation.Activation;
import org.brain4j.math.tensor.Tensor;

public class GraphConvLayer extends Layer {

    private int dimension;

    private GraphConvLayer() {
    }

    public GraphConvLayer(int dimension, Activation activation) {
        this.dimension = dimension;
        this.activation = activation;
    }

    public GraphConvLayer(int dimension, Activations activation) {
        this(dimension, activation.function());
    }

    public GraphConvLayer(int dimension) {
        this(dimension, Activations.LINEAR);
    }

    @Override
    public Layer connect(Layer previous) {
        this.weights = Tensors.zeros(previous.size(), dimension).withGrad();
        this.bias = Tensors.zeros(dimension).withGrad();
        return this;
    }

    @Override
    public Tensor[] forward(StatesCache cache, Tensor... inputs) {
        checkInputLength(2, inputs);

        Tensor features = inputs[0]; // [batch_size, nodes, features]
        Tensor adjacencyMatrix = inputs[1]; // [batch_size, nodes, nodes]

        // [batch_size, nodes, dimension]
        Tensor support = features.matmulGrad(weights);
        Tensor adjNorm = normalizeAdjacency(adjacencyMatrix);

        // [B, N, N] @ [B, N, D] -> [B, N, D]
        Tensor out = adjNorm.matmulGrad(support)
            .addGrad(bias);

        cache.rememberOutput(this, out);

        return new Tensor[] { out.activateGrad(activation), adjacencyMatrix };
    }

    @Override
    public int size() {
        return dimension;
    }
    
    @Override
    public void serialize(JsonObject object) {
        object.addProperty("dimension", dimension);
    }
    
    @Override
    public void deserialize(JsonObject object) {
        this.dimension = object.get("dimension").getAsInt();
    }
    
    @Override
    public boolean validInput(Tensor input) {
        return input.rank() == 3;
    }

    private Tensor normalizeAdjacency(Tensor adjacent) {
        int batch = adjacent.shape(0);
        int nodes = adjacent.shape(1);

        Tensor identity = Tensors.eye(nodes);

        // [batch_size, nodes, nodes]
        Tensor Ahat = adjacent.add(identity);
        // [batch_size, nodes]
        Tensor deg  = Ahat.sum(1, false).add(1e-12);
        Tensor degInvSqrt = deg.pow(-0.5);

        Tensor DinvSqrt = Tensors.zeros(batch, nodes, nodes);

        for (int b = 0; b < batch; b++) {
            for (int i = 0; i < nodes; i++) {
                double value = degInvSqrt.get(b, i);
                DinvSqrt.set(value, b, i, i);
            }
        }


        return DinvSqrt.matmulGrad(Ahat).matmulGrad(DinvSqrt);
    }
}
