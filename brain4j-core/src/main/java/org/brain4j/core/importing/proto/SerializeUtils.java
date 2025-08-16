package org.brain4j.core.importing.proto;

import org.brain4j.common.Tensors;
import org.brain4j.common.tensor.Tensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.function.Predicate;
import java.util.stream.Collectors;

public class SerializeUtils {
    
    public static List<org.brain4j.core.importing.proto.ProtoModel.Tensor> filterByName(List<org.brain4j.core.importing.proto.ProtoModel.Tensor> tensors, String name) {
        return filter(tensors, s -> s.getName().contains(name));
    }
    
    public static List<org.brain4j.core.importing.proto.ProtoModel.Tensor> filter(List<org.brain4j.core.importing.proto.ProtoModel.Tensor> tensors, Predicate<org.brain4j.core.importing.proto.ProtoModel.Tensor> predicate) {
        return tensors.stream().filter(predicate).toList();
    }
    
    public static Tensor deserializeTensor(org.brain4j.core.importing.proto.ProtoModel.Tensor tensor) {
        List<Float> rawData = tensor.getDataList();
        
        int[] shape = tensor.getShapeList().stream().mapToInt(Integer::intValue).toArray();
        float[] data = new float[rawData.size()];
        
        for (int i = 0; i < data.length; i++) {
            data[i] = rawData.get(i);
        }
        
        return Tensors.create(shape, data);
    }
    
    public static org.brain4j.core.importing.proto.ProtoModel.Tensor.Builder serializeTensor(String name, Tensor tensor) {
        if (tensor == null) tensor = Tensors.zeros(0);
        
        List<Integer> shape = Arrays.stream(tensor.shape()).boxed().collect(Collectors.toList());
        List<Float> data = new ArrayList<>();
        
        for (float val : tensor.data()) {
            data.add(val);
        }
        
        return org.brain4j.core.importing.proto.ProtoModel.Tensor.newBuilder()
            .setName(name)
            .addAllShape(shape)
            .addAllData(data);
    }
    
    public static float attribute(Map<String, org.brain4j.core.importing.proto.ProtoModel.AttrValue> attributes, String field, float defaultValue) {
        org.brain4j.core.importing.proto.ProtoModel.AttrValue value = attributes.get(field);
    
        if (value == null) return defaultValue;
        
        return value.getFloatVal();
    }
    
    public static float attribute(org.brain4j.core.importing.proto.ProtoModel.Layer layer, String field, float defaultValue) {
        return attribute(layer.getAttrsMap(), field, defaultValue);
    }
    
    public static double attribute(Map<String, org.brain4j.core.importing.proto.ProtoModel.AttrValue> attributes, String field, double defaultValue) {
        org.brain4j.core.importing.proto.ProtoModel.AttrValue value = attributes.get(field);
        
        if (value == null) return defaultValue;
        
        return value.getFloatVal();
    }
    
    public static double attribute(org.brain4j.core.importing.proto.ProtoModel.Layer layer, String field, double defaultValue) {
        return layer.getAttrsOrDefault(field, value(defaultValue)).getFloatVal();
    }
    
    public static int attribute(Map<String, org.brain4j.core.importing.proto.ProtoModel.AttrValue> attributes, String field, int defaultValue) {
        org.brain4j.core.importing.proto.ProtoModel.AttrValue value = attributes.get(field);
        
        if (value == null) return defaultValue;
        
        return value.getIntVal();
    }
    
    public static int attribute(org.brain4j.core.importing.proto.ProtoModel.Layer layer, String field, int defaultValue) {
        return layer.getAttrsOrDefault(field, value(defaultValue)).getIntVal();
    }
    
    public static String attribute(Map<String, org.brain4j.core.importing.proto.ProtoModel.AttrValue> attributes, String field, String defaultValue) {
        org.brain4j.core.importing.proto.ProtoModel.AttrValue value = attributes.get(field);
        
        if (value == null) return defaultValue;
        
        return value.getStringVal();
    }
    
    public static String attribute(org.brain4j.core.importing.proto.ProtoModel.Layer layer, String field, String defaultValue) {
        return layer.getAttrsOrDefault(field, value(defaultValue)).getStringVal();
    }
    
    public static org.brain4j.core.importing.proto.ProtoModel.AttrValue value(float field) {
        return org.brain4j.core.importing.proto.ProtoModel.AttrValue
            .newBuilder()
            .setFloatVal(field)
            .build();
    }
    
    public static org.brain4j.core.importing.proto.ProtoModel.AttrValue value(double field) {
        return org.brain4j.core.importing.proto.ProtoModel.AttrValue
            .newBuilder()
            .setFloatVal((float) field)
            .build();
    }
    
    public static org.brain4j.core.importing.proto.ProtoModel.AttrValue value(int field) {
        return org.brain4j.core.importing.proto.ProtoModel.AttrValue
            .newBuilder()
            .setIntVal(field)
            .build();
    }
    
    public static org.brain4j.core.importing.proto.ProtoModel.AttrValue value(String field) {
        return org.brain4j.core.importing.proto.ProtoModel.AttrValue
            .newBuilder()
            .setStringVal(field)
            .build();
    }
}
