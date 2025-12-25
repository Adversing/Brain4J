package org.brain4j.core.model;

import org.brain4j.core.layer.Layer;
import org.brain4j.core.model.impl.Sequential;

import java.util.ArrayList;
import java.util.List;

public class ModelSpecs implements ModelComponent {
    
    private final List<ModelComponent> components = new ArrayList<>();
    
    public static ModelSpecs of(ModelComponent... components) {
        if (components == null) throw new IllegalArgumentException("Component list cannot be null!");
        
        ModelSpecs specs = new ModelSpecs();
        specs.components.addAll(List.of(components));
        
        return specs;
    }
    
    @Override
    public void appendTo(List<Layer> layers) {
        for (ModelComponent component : components) {
            component.appendTo(layers);
        }
    }
    
    public ModelSpecs add(ModelComponent component) {
        components.add(component);
        return this;
    }
    
    public Model build() {
        return build(System.currentTimeMillis());
    }
    
    public Model build(long seed) {
        return new Sequential(this, null, seed);
    }
    
    public List<ModelComponent> getComponents() {
        return components;
    }
    
    public List<Layer> buildLayerList() {
        List<Layer> flat = new ArrayList<>();
        appendTo(flat);
        return flat;
    }
}
