package org.brain4j.core.importing;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import org.brain4j.math.Tensors;
import org.brain4j.math.tensor.Tensor;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Map;

import static org.brain4j.core.importing.impl.BrainFormat.GSON;

public class SafeTensorsConverter {

    public static final ByteOrder NATIVE_ORDER = ByteOrder.nativeOrder();
    
    public static byte[] save(Map<String, Tensor> weights) {
        JsonObject header = new JsonObject();
        int offset = 0;
        
        for (Map.Entry<String, Tensor> entry : weights.entrySet()) {
            String name = entry.getKey();
            Tensor weight = entry.getValue();
            
            int begin = offset;
            offset += weight.elements() * 4;
            int end = offset;
            
            JsonArray offsets = new JsonArray();
            offsets.add(begin);
            offsets.add(end);
            
            JsonArray shape = new JsonArray();
            
            for (int dimension : weight.shape()) {
                shape.add(dimension);
            }
            
            JsonObject tensor = new JsonObject();
            
            tensor.addProperty("dtype", "f32");
            tensor.add("shape", shape);
            tensor.add("offsets", offsets);
            
            header.add(name, tensor);
        }
        
        byte[] headerJson = GSON.toJson(header).getBytes();
        
        try (ByteArrayOutputStream stream = new ByteArrayOutputStream()) {
            ByteBuffer buffer = ByteBuffer.allocate(8)
                .order(NATIVE_ORDER)
                .putLong(headerJson.length);
            
            stream.write(buffer.array());
            stream.write(headerJson);
            
            for (Tensor tensor : weights.values()) {
                stream.write(tensor.toByteArray());
            }
            
            return stream.toByteArray();
        } catch (IOException e) {
            e.printStackTrace(System.err);
            return new byte[0];
        }
    }
    
    public static Map<String, Tensor> load(byte[] data) {
        Map<String, Tensor> weights = new HashMap<>();
        ByteBuffer buffer = ByteBuffer.wrap(data).order(NATIVE_ORDER);
        long headerLength = buffer.getLong();
        
        byte[] headerBytes = new byte[(int) headerLength];
        buffer.get(headerBytes);
        
        String headerJson = new String(headerBytes, StandardCharsets.UTF_8);
        JsonObject header = GSON.fromJson(headerJson, JsonObject.class);
        
        for (Map.Entry<String, JsonElement> entry : header.entrySet()) {
            String name = entry.getKey();
            JsonObject info = entry.getValue().getAsJsonObject();
            
            if (!info.has("shape")) continue;
            
            JsonArray shapeArray = info.getAsJsonArray("shape");
            
            int elements = 1;
            int[] shape = new int[shapeArray.size()];
            
            for (int i = 0; i < shapeArray.size(); i++) {
                shape[i] = shapeArray.get(i).getAsInt();
                elements *= shape[i];
            }
            
            JsonArray offsetArray = null;
            
            if (info.has("offsets")) offsetArray = info.getAsJsonArray("offsets");
            if (info.has("data_offsets")) offsetArray = info.getAsJsonArray("data_offsets");
            
            if (offsetArray == null) continue;
            
            int start = offsetArray.get(0).getAsInt();
            int end = offsetArray.get(1).getAsInt();
            
            int length = end - start;
            byte[] tensorBytes = new byte[length];
            
            buffer.position(8 + (int) headerLength + start);
            buffer.get(tensorBytes);
            
            float[] values = new float[elements];

            ByteBuffer tensorBuffer = ByteBuffer.wrap(tensorBytes).order(ByteOrder.LITTLE_ENDIAN);
            tensorBuffer.asFloatBuffer().get(values);

            Tensor tensor = Tensors.create(shape, values);
            weights.put(name, tensor);
        }
        
        return weights;
    }
}
