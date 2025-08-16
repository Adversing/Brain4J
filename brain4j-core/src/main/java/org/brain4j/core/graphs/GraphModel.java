package org.brain4j.core.graphs;

import org.brain4j.common.data.ListDataSource;
import org.brain4j.common.gpu.GpuContext;
import org.brain4j.common.gpu.device.Device;
import org.brain4j.common.tensor.Tensor;
import org.brain4j.core.layer.Layer;
import org.brain4j.core.loss.LossFunction;
import org.brain4j.core.model.Model;
import org.brain4j.core.training.StatesCache;
import org.brain4j.core.training.optimizer.Optimizer;
import org.brain4j.core.training.updater.Updater;
import org.brain4j.core.training.wrappers.EvaluationResult;

import java.util.*;

public class GraphModel implements Model {

    private final List<GraphNode> nodes;
    private final List<String> inputNames;
    private final List<String> outputNames;
    private final Map<String, Tensor> initializers;

    private Device device;

    public GraphModel(
        List<GraphNode> nodes,
        List<String> inputNames,
        List<String> outputNames,
        Map<String, Tensor> initializers
    ) {
        this.nodes = nodes;
        this.inputNames = inputNames;
        this.outputNames = outputNames;
        this.initializers = initializers;
    }

    public static Builder newGraph() {
        return new Builder();
    }

    @Override
    public Tensor[] predict(StatesCache cache, boolean training, Tensor... inputs) {
        if (inputs.length != inputNames.size()) {
            throw new IllegalArgumentException("Expected " + inputNames.size() + " inputs, but got " + inputs.length);
        }
        
        if (device != null) {
            GpuContext.updateQueue(device, cache.commandQueue());
        }

        Map<String, Tensor> computed = new HashMap<>(initializers);

        for (int i = 0; i < inputs.length; i++) {
            computed.put(inputNames.get(i), inputs[i].to(device));
        }

        for (GraphNode node : nodes) {
            List<String> inputNames = node.inputs();
            Tensor[] inputTensors = new Tensor[inputNames.size()];

            for (int j = 0; j < inputTensors.length; j++) {
                Tensor input = computed.get(inputNames.get(j));

                if (input == null) {
                    throw new IllegalStateException("Missing tensor for input: " + inputNames.get(j));
                }

                inputTensors[j] = input.to(device);
            }

            Tensor output = node.operation().compute(inputTensors);

            for (String outputName : node.outputs()) {
                computed.put(outputName, output);
            }
        }
        
        if (!training && device != null) {
            GpuContext.closeQueue(device);
        }

        Tensor[] outputs = new Tensor[outputNames.size()];

        for (int i = 0; i < outputs.length; i++) {
            outputs[i] = computed.get(outputNames.get(i));
        }

        return outputs;
    }

    @Override
    public void backpropagate(StatesCache cache, Tensor[] outputs, Tensor[] targets) {
        throw new UnsupportedOperationException();
    }

    @Override
    public Model add(Layer layer) {
        throw new UnsupportedOperationException();
    }

    @Override
    public Model add(int index, Layer layer) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void fit(ListDataSource train, ListDataSource validation, int epoches, int evaluateEvery) {
        throw new UnsupportedOperationException();
    }

    @Override
    public EvaluationResult evaluate(ListDataSource dataSource) {
        return null;
    }

    @Override
    public double loss(ListDataSource dataSource) {
        return 0;
    }

    @Override
    public Model compile(LossFunction lossFunction, Optimizer optimizer, Updater updater) {
        throw new UnsupportedOperationException();
    }

    @Override
    public Model to(Device device) {
        this.device = device;

        Map<String, Tensor> copy = new HashMap<>(initializers);

        initializers.clear();

        for (Map.Entry<String, Tensor> entry : copy.entrySet()) {
            Tensor weight = entry.getValue().to(device);
            initializers.put(entry.getKey(), weight);
        }

        return this;
    }

    @Override
    public Device device() {
        return device;
    }

    @Override
    public List<Layer> layers() {
        throw new UnsupportedOperationException();
    }

    @Override
    public List<Layer> flattened() {
        throw new UnsupportedOperationException();
    }

    @Override
    public Layer layerAt(int index) {
        throw new UnsupportedOperationException();
    }

    @Override
    public Layer flattenedAt(int index) {
        throw new UnsupportedOperationException();
    }
    
    @Override
    public Optimizer optimizer() {
        throw new UnsupportedOperationException();
    }
    
    @Override
    public void setOptimizer(Optimizer optimizer) {
        throw new UnsupportedOperationException();
    }
    
    @Override
    public Updater updater() {
        throw new UnsupportedOperationException();
    }
    
    @Override
    public void setUpdater(Updater updater) {
        throw new UnsupportedOperationException();
    }
    
    @Override
    public LossFunction lossFunction() {
        throw new UnsupportedOperationException();
    }
    
    @Override
    public void setLossFunction(LossFunction lossFunction) {
        throw new UnsupportedOperationException();
    }
    
    @Override
    public void summary() {
        throw new UnsupportedOperationException();
    }

    @Override
    public void zeroGrad() {
        throw new UnsupportedOperationException();
    }

    @Override
    public Iterator<Layer> iterator() {
        throw new UnsupportedOperationException();
    }

    public static class Builder {

        private final List<GraphNode> nodes = new ArrayList<>();
        private final Map<String, Tensor> initializers = new HashMap<>();
        private List<String> inputs = new ArrayList<>();
        private List<String> outputs = new ArrayList<>();

        public Builder addNode(GraphNode node) {
            this.nodes.add(node);
            return this;
        }

        public Builder initializer(String name, Tensor tensor) {
            this.initializers.put(name, tensor);
            return this;
        }

        public Builder inputs(List<String> inputs) {
            this.inputs = inputs;
            return this;
        }

        public Builder outputs(List<String> outputs) {
            this.outputs = outputs;
            return this;
        }

        public GraphModel compile() {
            return new GraphModel(nodes, inputs, outputs, initializers);
        }
    }
}
