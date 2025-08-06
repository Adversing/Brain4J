package org.brain4j.core.importing.impl;

import org.brain4j.core.importing.ModelLoader;
import org.brain4j.core.importing.proto.ProtoModel;
import org.brain4j.core.layer.Layer;
import org.brain4j.core.model.Model;

import java.io.*;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;

public class BrainProtocol implements ModelLoader {
    
    @Override
    public Model deserialize(byte[] bytes) {
        return null;
    }
    
    @Override
    public void serialize(Model model, File file) throws IOException {
        ByteArrayOutputStream stream = new ByteArrayOutputStream();
        
        ProtoModel.Model.Builder builder =
            ProtoModel.Model.newBuilder()
                .setVersion(1)
                .setName(file.getName())
                .setCreated(Instant.now().toString());
        
        List<Layer> layers = model.layers();
        
        for (int i = 0; i < layers.size(); i++) {
            Layer layer = layers.get(i);
            String name = layer.getClass().getSimpleName().toLowerCase();
            String id = name + "." + i;
            
            ProtoModel.Layer.Builder layerBuilder =
                ProtoModel.Layer.newBuilder()
                    .setName(id)
                    .setType(layer.getClass().getSimpleName())
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
