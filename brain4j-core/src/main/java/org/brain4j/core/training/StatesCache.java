package org.brain4j.core.training;

import org.brain4j.core.transformer.attention.head.AttentionHead;
import org.brain4j.math.gpu.device.Device;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.core.layer.Layer;
import org.jocl.cl_command_queue;

import java.util.HashMap;
import java.util.Map;

public class StatesCache {

    private final Map<Layer, Tensor[]> inputStates;
    private final Map<Layer, Tensor[]> outputStates;
    private final boolean training;
    private cl_command_queue commandQueue;

    // LLM KV cache
    private Map<AttentionHead, Tensor> keys;
    private Map<AttentionHead, Tensor> values;
    private Map<AttentionHead, Tensor> attentionOutput;

    public StatesCache() {
        this(false, null);
    }

    public StatesCache(boolean training, Device device) {
        this.training = training;
        this.inputStates = new HashMap<>();
        this.outputStates = new HashMap<>();
        this.keys = new HashMap<>();
        this.values = new HashMap<>();
        this.attentionOutput = new HashMap<>();

        if (device != null) {
            this.commandQueue = device.newCommandQueue();
        }
    }
    
    public boolean training() {
        return training;
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

    public Tensor keys(AttentionHead head) {
        return keys.get(head);
    }

    public void setKeys(AttentionHead head, Tensor keys) {
        this.keys.put(head, keys);
    }

    public Tensor values(AttentionHead head) {
        return values.get(head);
    }

    public void setValues(AttentionHead head, Tensor values) {
        this.values.put(head, values);
    }

    public void setAttentionOutput(AttentionHead head, Tensor attentionOutput) {
        this.attentionOutput.put(head, attentionOutput);
    }

    public Tensor attentionOutput(AttentionHead head) {
        return attentionOutput.get(head);
    }
}

