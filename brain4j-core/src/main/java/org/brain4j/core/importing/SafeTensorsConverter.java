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
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
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
    
    public static Map<String, Tensor> load(Path path) throws IOException {
        try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
            ByteBuffer headerLenBuf = ByteBuffer.allocate(8).order(NATIVE_ORDER);
            int readHeaderLen = 0;
            
            while (headerLenBuf.hasRemaining()) {
                readHeaderLen += channel.read(headerLenBuf, readHeaderLen);
            }
            
            headerLenBuf.rewind();
            long headerLengthLong = headerLenBuf.getLong();
            
            if (headerLengthLong > Integer.MAX_VALUE) {
                throw new IOException("Header too large (>2GB) unexpected for safetensors header");
            }
            
            int headerLength = (int) headerLengthLong;
            ByteBuffer headerBuf = ByteBuffer.allocate(headerLength);
            int read = 0;
            
            while (headerBuf.hasRemaining()) {
                int r = channel.read(headerBuf, 8 + read);
                if (r < 0) throw new IOException("Unexpected EOF while reading header");
                read += r;
            }
            
            headerBuf.rewind();
            
            String headerJson = StandardCharsets.UTF_8.decode(headerBuf).toString();
            JsonObject header = GSON.fromJson(headerJson, JsonObject.class);
            
            Map<String, Tensor> weights = new HashMap<>();
            long baseDataOffset = 8L + headerLength;
            
            for (Map.Entry<String, JsonElement> entry : header.entrySet()) {
                String name = entry.getKey();
                JsonObject info = entry.getValue().getAsJsonObject();
                
                if (!info.has("shape")) continue;
                
                JsonArray shapeArray = info.getAsJsonArray("shape");
                
                int[] shape = new int[shapeArray.size()];
                int elements = 1;
                
                for (int i = 0; i < shapeArray.size(); i++) {
                    shape[i] = shapeArray.get(i).getAsInt();
                    elements *= shape[i];
                }
                
                JsonArray offsets = info.has("offsets")
                    ? info.getAsJsonArray("offsets")
                    : info.has("data_offsets")
                    ? info.getAsJsonArray("data_offsets") : null;
                
                if (offsets == null || offsets.size() < 2) continue;
                
                long start = offsets.get(0).getAsLong();
                long end = offsets.get(1).getAsLong();
                long length = end - start;
                
                if (length < 0) throw new IOException("Invalid tensor offsets for " + name);
                
                long absolutePos = baseDataOffset + start;
                
                float[] values = new float[elements];
                MappedByteBuffer mb = channel.map(FileChannel.MapMode.READ_ONLY, absolutePos, length);
                mb.order(ByteOrder.LITTLE_ENDIAN);
                mb.asFloatBuffer().get(values);
                
                weights.put(name, Tensors.create(shape, values));
            }
            
            return weights;
        }
    }
}
