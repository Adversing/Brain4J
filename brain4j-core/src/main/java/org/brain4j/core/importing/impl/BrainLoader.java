package org.brain4j.core.importing.impl;

import org.brain4j.common.Commons;
import org.brain4j.core.importing.ModelLoader;
import org.brain4j.core.importing.proto.ProtoModel;
import org.brain4j.core.layer.Layer;
import org.brain4j.core.loss.LossFunction;
import org.brain4j.core.model.Model;
import org.brain4j.core.model.impl.Sequential;

import java.io.*;
import java.lang.reflect.Constructor;
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
            
            Class<?> clazz = Class.forName(layerType);
            
            Constructor<?> constructor = clazz.getDeclaredConstructor();
            constructor.setAccessible(true);
            
            Layer wrapped = (Layer) constructor.newInstance();
            List<ProtoModel.Tensor> tensors = new ArrayList<>();
            
            for (ProtoModel.Tensor tensor : protoModel.getWeightsList()) {
                if (!tensor.getName().startsWith(layerId)) continue;
                
                tensors.add(tensor);
            }
            
            positionMap.put(position, wrapped);
            wrapped.deserialize(tensors, layer);
        }
        
        List<Integer> positions = new ArrayList<>(positionMap.keySet());
        Collections.sort(positions);
        
        Sequential model = Sequential.of();
        
        for (int pos : positions) {
            model.add(positionMap.get(pos));
        }
        
        String lossFunctionClass = protoModel.getLossFunction();
        
        LossFunction function = Commons.newInstance(lossFunctionClass);
        model.setLossFunction(function);
        
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
            
            ProtoModel.Layer.Builder layerBuilder =
                ProtoModel.Layer.newBuilder()
                    .setName(id)
                    .setType(layer.getClass().getName())
                    .setDimension(layer.size());
            
            List<ProtoModel.Tensor.Builder> tensorsBuilders = layer.serialize(layerBuilder);
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
}
