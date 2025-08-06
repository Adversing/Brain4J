package org.brain4j.core.importing.proto;

import org.brain4j.common.Tensors;
import org.brain4j.common.tensor.Tensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

public interface SerializationInstance {
    
    default Tensor deserializeTensor(ProtoModel.Tensor tensor) {
        List<Float> rawData = tensor.getDataList();
        
        int[] shape = tensor.getShapeList().stream().mapToInt(Integer::intValue).toArray();
        float[] data = new float[rawData.size()];
        
        for (int i = 0; i < data.length; i++) {
            data[i] = rawData.get(i);
        }
        
        return Tensors.create(shape, data);
    }
    
    default ProtoModel.Tensor.Builder serializeTensor(String name, Tensor tensor) {
        List<Integer> shape = Arrays.stream(tensor.shape()).boxed().collect(Collectors.toList());
        List<Float> data = new ArrayList<>();
        
        for (float val : tensor.data()) {
            data.add(val);
        }
        
        return ProtoModel.Tensor.newBuilder()
            .setName(name)
            .addAllShape(shape)
            .addAllData(data);
    }
    
    default float attribute(ProtoModel.Layer layer, String field, float defaultValue) {
        return layer.getAttrsOrDefault(field, value(defaultValue)).getFloatVal();
    }
    
    default double attribute(ProtoModel.Layer layer, String field, double defaultValue) {
        return layer.getAttrsOrDefault(field, value(defaultValue)).getFloatVal();
    }
    
    default int attribute(ProtoModel.Layer layer, String field, int defaultValue) {
        return layer.getAttrsOrDefault(field, value(defaultValue)).getIntVal();
    }
    
    default String attribute(ProtoModel.Layer layer, String field, String defaultValue) {
        return layer.getAttrsOrDefault(field, value(defaultValue)).getStringVal();
    }
    
    default ProtoModel.AttrValue value(float field) {
        return ProtoModel.AttrValue
            .newBuilder()
            .setFloatVal(field)
            .build();
    }
    
    default ProtoModel.AttrValue value(double field) {
        return ProtoModel.AttrValue
            .newBuilder()
            .setFloatVal((float) field)
            .build();
    }
    
    default ProtoModel.AttrValue value(int field) {
        return ProtoModel.AttrValue
            .newBuilder()
            .setIntVal(field)
            .build();
    }
    
    default ProtoModel.AttrValue value(String field) {
        return ProtoModel.AttrValue
            .newBuilder()
            .setStringVal(field)
            .build();
    }
}
