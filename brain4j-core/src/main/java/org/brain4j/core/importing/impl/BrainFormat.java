package org.brain4j.core.importing.impl;

import org.brain4j.math.Commons;
import org.brain4j.core.importing.ModelFormat;
import org.brain4j.core.importing.proto.ProtoModel;
import org.brain4j.core.importing.registry.LayerRegistry;
import org.brain4j.core.layer.Layer;
import org.brain4j.core.layer.impl.transformer.TransformerEncoder;
import org.brain4j.core.model.Model;

import java.io.*;
import java.time.Instant;
import java.util.*;
import java.util.function.Supplier;

public class BrainFormat implements ModelFormat {

    private final LayerRegistry registry = new LayerRegistry();

    public BrainFormat() {
        registry.registerAll(LAYER_IDENTITY_MAP);
    }

    @Override
    public <T extends Model> T deserialize(byte[] bytes, Supplier<T> constructor) throws Exception {
        ProtoModel.Model protoModel = ProtoModel.Model.parseFrom(bytes);
        Map<Integer, Layer> positionMap = new HashMap<>();
        
        for (ProtoModel.Layer layer : protoModel.getLayersList()) {
            String layerType = layer.getType();
            String layerId = layer.getName();
            
            String[] parts = layerId.split("\\.");
            
            if (parts.length == 0) {
                throw new IllegalArgumentException("Layer does not match format!");
            }
            
            int position = Integer.parseInt(parts[1]);
            Layer wrapped = registry.create(layerType);

            List<ProtoModel.Tensor> tensors = protoModel.getWeightsList().stream()
                .filter(t -> t.getName().startsWith(layerId))
                .toList();
            
            wrapped.deserialize(tensors, layer);
            positionMap.put(position, wrapped);
        }
        
        T model = constructor.get();
        model.setLossFunction(Commons.newInstance(protoModel.getLossFunction()));
        
        positionMap.entrySet().stream()
            .sorted(Map.Entry.comparingByKey())
            .forEach(e -> model.add(e.getValue()));
        
        return model;
    }
    
    @Override
    public void serialize(Model model, File file) throws IOException {
        ProtoModel.Model.Builder builder =
            ProtoModel.Model.newBuilder()
                .setVersion(1)
                .setName(file.getName())
                .setCreated(Instant.now().toString())
                .setLossFunction(model.lossFunction().getClass().getName());
        
        List<Layer> layers = model.flattened();
        
        for (int i = 0; i < layers.size(); i++) {
            Layer layer = layers.get(i);
            
            String name = layer.getClass().getSimpleName().toLowerCase();
            String id = name + "." + i;
            
            ProtoModel.Layer.Builder layerBuilder = toProtoBuilder(layer).setName(id);
            layer.serialize(layerBuilder);
            
            List<ProtoModel.Tensor.Builder> tensorsBuilders = layer.weightsList();
            List<ProtoModel.Tensor> tensors = new ArrayList<>();
            
            for (ProtoModel.Tensor.Builder tensorBuilder : tensorsBuilders) {
                tensorBuilder.setName(id + "." + tensorBuilder.getName());
                tensors.add(tensorBuilder.build());
            }
            
            builder.addLayers(layerBuilder.build());
            builder.addAllWeights(tensors);
        }
        
        builder.build().writeTo(new FileOutputStream(file));
    }

    public LayerRegistry registry() {
        return registry;
    }

    public ProtoModel.Layer.Builder toProtoBuilder(Layer layer) {
        String identifier = registry.fromState(layer.getClass());

        if (identifier == null) {
            throw new IllegalArgumentException(
                "Unable to map layer " + layer.getClass().getName() + ". If you are using " +
                "a custom layer, make sure to register it first."
            );
        }

        ProtoModel.Layer.Builder builder = ProtoModel.Layer.newBuilder()
            .setType(identifier);
        
        if (!(layer instanceof TransformerEncoder)) {
            builder.setBasic(
                ProtoModel.BasicLayer.newBuilder()
                    .setDimension(layer.size())
            );
        }
        
        return builder;
    }
}
