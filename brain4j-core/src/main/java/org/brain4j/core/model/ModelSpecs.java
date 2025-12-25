package org.brain4j.core.model;

import org.brain4j.core.layer.Layer;

import java.util.ArrayList;
import java.util.List;

public class ModelSpecs {

    private final List<Layer> layers = new ArrayList<>();

    public static ModelSpecs of(Layer... layers) {
        ModelSpecs specs = new ModelSpecs();
        specs.layers.addAll(List.of(layers));
        return specs;
    }

    public ModelSpecs add(Layer layer) {
        layers.add(layer);
        return this;
    }

    public Model build() {
        return build(System.currentTimeMillis());
    }

    public Model build(long seed) {
        return null; // TODO
    }
}
