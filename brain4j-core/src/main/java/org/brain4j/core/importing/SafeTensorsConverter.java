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
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;

import static org.brain4j.core.importing.format.impl.BrainFormat.GSON;

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
            tensor.add("data_offsets", offsets);
            
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
    
    public static Map<String, Tensor> load(Path path) throws IOException {
        byte[] data = Files.readAllBytes(path);
        return load(data);
    }
    
    public static Map<String, Tensor> load(byte[] data) throws IOException {
        ByteBuffer buffer = ByteBuffer.wrap(data).order(NATIVE_ORDER);
        return load(buffer);
    }
    
    private static Map<String, Tensor> load(ByteBuffer buffer) throws IOException {
        buffer.order(NATIVE_ORDER);
        
        if (buffer.remaining() < 8) {
            throw new IOException("Invalid safetensors buffer");
        }
        
        long headerLengthLong = buffer.getLong();
        
        if (headerLengthLong > Integer.MAX_VALUE) {
            throw new IOException("Header too large (>2GB)");
        }
        
        int headerLength = (int) headerLengthLong;
        
        if (buffer.remaining() < headerLength) {
            throw new IOException("Unexpected EOF while reading header");
        }
        
        byte[] headerBytes = new byte[headerLength];
        buffer.get(headerBytes);
        
        String headerJson = new String(headerBytes, StandardCharsets.UTF_8);
        JsonObject header = GSON.fromJson(headerJson, JsonObject.class);
        
        long baseDataOffset = 8L + headerLength;
        Map<String, Tensor> weights = new HashMap<>();
        
        for (Map.Entry<String, JsonElement> entry : header.entrySet()) {
            String name = entry.getKey();
            JsonObject info = entry.getValue().getAsJsonObject();
            
            JsonArray shapeArray = info.getAsJsonArray("shape");
            int[] shape = new int[shapeArray.size()];
            int elements = 1;
            
            for (int i = 0; i < shape.length; i++) {
                shape[i] = shapeArray.get(i).getAsInt();
                elements *= shape[i];
            }
            
            JsonArray offsets = info.has("offsets")
                ? info.getAsJsonArray("offsets")
                : info.getAsJsonArray("data_offsets");
            
            long start = offsets.get(0).getAsLong();
            long end   = offsets.get(1).getAsLong();
            
            int byteLength = Math.toIntExact(end - start);
            
            int pos = Math.toIntExact(baseDataOffset + start);
            ByteBuffer slice = buffer.duplicate();
            slice.position(pos);
            slice.limit(pos + byteLength);
            slice = slice.slice().order(ByteOrder.LITTLE_ENDIAN);
            
            float[] values = new float[elements];
            slice.asFloatBuffer().get(values);
            
            weights.put(name, Tensors.create(shape, values));
        }
        
        return weights;
    }
}
