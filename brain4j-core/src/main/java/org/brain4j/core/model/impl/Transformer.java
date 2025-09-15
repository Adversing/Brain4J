package org.brain4j.core.model.impl;

import org.brain4j.core.layer.Layer;
import org.brain4j.core.model.Model;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.gpu.GpuContext;
import org.brain4j.math.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;

public class Transformer extends Sequential {
    
    public static Transformer of(Layer... layers) {
        return new Transformer(layers);
    }
    
    protected Transformer(Layer... layers) {
        this.layers = new ArrayList<>(List.of(layers));
        this.flattened = new ArrayList<>();
        this.seed = System.currentTimeMillis();
        
        for (Layer layer : layers) {
            if (layer instanceof Model subModel) {
                flattened.addAll(subModel.flattened());
                continue;
            }
            
            flattened.add(layer);
        }
    }
    
    @Override
    public Tensor[] predict(StatesCache cache, Tensor... inputs) {
        if (flattened.isEmpty()) {
            throw new IllegalArgumentException("No layers found!");
        }
        
        Tensor[] validated = validateInputs(inputs);
        Tensor[] result = new Tensor[validated.length];
        
        for (int i = 0; i < validated.length; i++) {
            result[i] = validated[i].to(device).withGrad();
        }
        
        if (device != null) {
            GpuContext.updateQueue(device, cache);
        }
        
        for (Layer layer : flattened) {
            result = layer.forward(cache, result);
        }
        
        if (!cache.training() && device != null) {
            GpuContext.closeQueue(device);
        }
        
        return result;
    }
}
