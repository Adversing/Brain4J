package org.brain4j.core.training;

import org.brain4j.core.transformer.attention.MultiHeadAttention;
import org.brain4j.core.transformer.attention.head.AttentionHead;
import org.brain4j.math.gpu.device.Device;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.core.layer.Layer;
import org.jocl.cl_command_queue;

import java.util.HashMap;
import java.util.Map;

public class StatesCache {

    private final Map<Object, Tensor> tensorCache;
    private final Map<Layer, Tensor[]> inputStates;
    private final Map<Layer, Tensor[]> outputStates;
    private final boolean training;
    private cl_command_queue commandQueue;

    public StatesCache() {
        this(false, null);
    }

    public StatesCache(boolean training, Device device) {
        this.training = training;
        this.inputStates = new HashMap<>();
        this.outputStates = new HashMap<>();
        this.tensorCache = new HashMap<>();

        if (device != null) {
            this.commandQueue = device.newCommandQueue();
        }
    }
    
    public boolean training() {
        return training;
    }

    public Tensor get(Object key) {
        return tensorCache.get(key);
    }

    public void updateCache(Object key, Tensor value) {
        tensorCache.put(key, value);
    }
    
    public Tensor[] input(Layer layer) {
        return inputStates.get(layer);
    }

    public void rememberInput(Layer layer, Tensor... tensor) {
        inputStates.put(layer, tensor);
    }

    public Tensor[] output(Layer layer) {
        return outputStates.computeIfAbsent(layer, (l) -> new Tensor[0]);
    }

    public void rememberOutput(Layer layer, Tensor... state) {
        outputStates.put(layer, state);
    }
    
    public cl_command_queue commandQueue() {
        return commandQueue;
    }
}

