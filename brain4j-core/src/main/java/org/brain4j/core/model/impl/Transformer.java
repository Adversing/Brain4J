package org.brain4j.core.model.impl;

import org.brain4j.core.layer.Layer;
import org.brain4j.core.layer.impl.convolutional.InputLayer;
import org.brain4j.core.model.Model;
import org.brain4j.core.training.StatesCache;
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
}
