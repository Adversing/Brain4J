package org.brain4j.core.importing.impl;

import org.brain4j.common.Commons;
import org.brain4j.core.importing.ModelLoader;
import org.brain4j.core.importing.proto.ProtoModel;
import org.brain4j.core.layer.Layer;
import org.brain4j.core.layer.impl.transformer.TransformerEncoder;
import org.brain4j.core.model.Model;
import org.brain4j.core.model.impl.Sequential;
import org.brain4j.core.transformer.attention.MultiHeadAttention;

import java.io.*;
import java.time.Instant;
import java.util.*;

public class BrainLoader implements ModelLoader {
    
    @Override
    public Model deserialize(byte[] bytes) throws Exception {
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
            Layer wrapped = Commons.newInstance(layerType);
            
            List<ProtoModel.Tensor> tensors = protoModel.getWeightsList().stream()
                .filter(t -> t.getName().startsWith(layerId))
                .toList();
            
            wrapped.deserialize(tensors, layer);
            positionMap.put(position, wrapped);
        }
        
        Sequential model = Sequential.of();
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
        
        List<Layer> layers = model.layers();
        
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
    
    public ProtoModel.Layer.Builder toProtoBuilder(Layer layer) {
        ProtoModel.Layer.Builder builder = ProtoModel.Layer.newBuilder()
            .setType(layer.getClass().getName());
        
        if (layer instanceof TransformerEncoder transformer) {
            // TODO: Finish implementation
            ProtoModel.Transformer.Builder transformerBuilder =
                ProtoModel.Transformer.newBuilder();
            
            List<Layer> subLayers = transformer.subLayers();
            
            for (Layer sub : subLayers) {
                transformerBuilder.addSubLayers(toProtoBuilder(sub));
            }
            
            builder.setTransformer(transformerBuilder);
        } else {
            builder.setBasic(
                ProtoModel.BasicLayer.newBuilder()
                    .setDimension(layer.size())
            );
        }
        
        return builder;
    }
}
