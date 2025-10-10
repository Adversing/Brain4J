package org.brain4j.core.importing.impl;

import com.google.gson.*;
import org.brain4j.core.importing.format.ModelFormat;
import org.brain4j.core.layer.Layer;
import org.brain4j.core.loss.LossFunction;
import org.brain4j.core.model.Model;
import org.brain4j.core.model.impl.Sequential;
import org.brain4j.core.training.optimizer.Optimizer;
import org.brain4j.core.training.updater.Updater;
import org.brain4j.math.Tensors;
import org.brain4j.math.activation.Activation;
import org.brain4j.math.clipper.GradientClipper;
import org.brain4j.math.tensor.Tensor;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.time.Instant;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;
import java.util.zip.Deflater;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;
import java.util.zip.ZipOutputStream;

import static org.brain4j.core.importing.Registries.*;

public class BrainFormat implements ModelFormat {
    
    public static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();
    public static final int FORMAT_VERSION = 1;

    @Override
    public Sequential deserialize(File file) {
        Sequential model = Sequential.of();
        Map<String, byte[]> files = new HashMap<>();
        
        try (FileInputStream stream = new FileInputStream(file);
             ZipInputStream zis = new ZipInputStream(stream)) {
            
            ZipEntry entry;
            
            while ((entry = zis.getNextEntry()) != null) {
                ByteArrayOutputStream baos = new ByteArrayOutputStream();
                
                byte[] buffer = new byte[8192];
                int read;
                
                while ((read = zis.read(buffer)) != -1) {
                    baos.write(buffer, 0, read);
                }
                
                files.put(entry.getName(), baos.toByteArray());
                zis.closeEntry();
            }
        } catch (Exception e) {
            e.printStackTrace(System.err);
        }
        
        if (files.containsKey("metadata.json")) {
            deserializeMetadata(model, files.get("metadata.json"));
        }
        
        if (files.containsKey("weights.safetensors")) {
            deserializeWeights(model, files.get("weights.safetensors"));
        }
        
        return model;
    }
    
    @Override
    public void serialize(Model model, File file) {
        Map<String, Tensor> globalWeightsMap = new HashMap<>();
        
        byte[] metadata = buildMetadata(model, globalWeightsMap);
        byte[] weights = buildWeights(globalWeightsMap);
        
        // TODO: serialize training metadata
        String[] files = { "metadata.json", "weights.safetensors" };
        byte[][] content = { metadata, weights };
        
        try (ByteArrayOutputStream stream = new ByteArrayOutputStream();
             ZipOutputStream zos = new ZipOutputStream(stream)) {
            zos.setLevel(Deflater.BEST_COMPRESSION);
            
            for (int i = 0; i < files.length; i++) {
                ZipEntry metaEntry = new ZipEntry(files[i]);
                byte[] fileContent = content[i];
                
                metaEntry.setMethod(ZipEntry.DEFLATED);
                metaEntry.setSize(fileContent.length);
                
                zos.putNextEntry(metaEntry);
                zos.write(fileContent);
            }
            
            zos.closeEntry();
            zos.close();
            
            byte[] zipData = stream.toByteArray();
            
            try (FileOutputStream fos = new FileOutputStream(file)) {
                fos.write(zipData);
            }
        } catch (IOException e) {
            e.printStackTrace(System.err);
        }
    }
    
    private <T extends Model> void deserializeMetadata(T model, byte[] data) {
        try {
            String json = new String(data, StandardCharsets.UTF_8);
            JsonObject metadata = GSON.fromJson(json, JsonObject.class);
            
            JsonArray architecture = metadata.getAsJsonArray("architecture");
            
            String optimizerId = metadata.get("optimizer").getAsString();
            String lossFunctionId = metadata.get("loss_function").getAsString();
            String updaterId = metadata.get("updater").getAsString();
            
            model.setOptimizer(OPTIMIZERS_REGISTRY.toInstance(optimizerId));
            model.setLossFunction(LOSS_FUNCTION_REGISTRY.toInstance(lossFunctionId));
            model.setUpdater(UPDATERS_REGISTRY.toInstance(updaterId));
            
            for (int i = 0; i < architecture.size(); i++) {
                JsonObject layerJson = architecture.get(i).getAsJsonObject();
                
                String type = layerJson.get("type").getAsString();
                String activationId = layerJson.get("activation").getAsString();
                String clipperId = layerJson.get("clipper").getAsString();
                
                Layer layer = LAYER_REGISTRY.toInstance(type);
                layer.activation(ACTIVATION_REGISTRY.toInstance(activationId));
                layer.clipper(CLIPPERS_REGISTRY.toInstance(clipperId));
                
                layer.deserialize(layerJson);
                model.add(layer);
            }
        } catch (Exception e) {
            e.printStackTrace(System.err);
        }
    }
    
    private <T extends Model> void deserializeWeights(T model, byte[] data) {
        try {
            Map<Layer, Map<String, Tensor>> weightsMap = new HashMap<>();
            ByteBuffer buffer = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);
            
            long headerLength = buffer.getLong();
            byte[] headerBytes = new byte[(int) headerLength];
            buffer.get(headerBytes);
            
            String headerJson = new String(headerBytes, StandardCharsets.UTF_8);
            JsonObject header = GSON.fromJson(headerJson, JsonObject.class);
            
            for (Map.Entry<String, JsonElement> entry : header.entrySet()) {
                String name = entry.getKey();
                
                JsonObject info = entry.getValue().getAsJsonObject();
                JsonArray shapeArray = info.getAsJsonArray("shape");
                
                int elements = 1;
                int[] shape = new int[shapeArray.size()];
                
                for (int i = 0; i < shapeArray.size(); i++) {
                    shape[i] = shapeArray.get(i).getAsInt();
                    elements *= shape[i];
                }
                
                JsonArray offsetArray = info.getAsJsonArray("offset");
                int start = offsetArray.get(0).getAsInt();
                int end = offsetArray.get(1).getAsInt();
                int length = end - start;
                
                byte[] tensorBytes = new byte[length];
                
                buffer.position(8 + (int) headerLength + start);
                buffer.get(tensorBytes);
                
                float[] values = new float[elements];
                ByteBuffer tensorBuffer = ByteBuffer.wrap(tensorBytes).order(ByteOrder.LITTLE_ENDIAN);
                
                for (int i = 0; i < elements; i++) {
                    values[i] = tensorBuffer.getFloat();
                }
                
                Tensor tensor = Tensors.create(shape, values);
                String[] parts = name.split("\\.");
                
                int layerIndex = Integer.parseInt(parts[1]);
                
                Layer layer = model.flattened().get(layerIndex);
                
                Map<String, Tensor> weights = weightsMap.computeIfAbsent(layer, (l) -> new HashMap<>());
                String weightName = parts[2];
                
                int nameIndex = name.indexOf(weightName);
                weights.put(name.substring(nameIndex), tensor);
            }
            
            for (Map.Entry<Layer, Map<String, Tensor>> entry : weightsMap.entrySet()) {
                Layer layer = entry.getKey();
                layer.loadWeights(entry.getValue());
            }
        } catch (Exception e) {
            e.printStackTrace(System.err);
        }
    }
    
    private byte[] buildMetadata(Model model, Map<String, Tensor> globalWeightsMap) {
        JsonObject metadata = new JsonObject();
        Instant date = Instant.now();
        
        Class<? extends Optimizer> optimizerClass = model.optimizer().getClass();
        Class<? extends LossFunction> lossFunctionClass = model.lossFunction().getClass();
        Class<? extends Updater> updaterClass = model.updater().getClass();
        
        metadata.addProperty("format_version", FORMAT_VERSION);
        metadata.addProperty("created_at", date.toString());
        metadata.addProperty("weights_file", "weights.safetensors");
        
        metadata.addProperty("optimizer", OPTIMIZERS_REGISTRY.fromClass(optimizerClass));
        metadata.addProperty("loss_function", LOSS_FUNCTION_REGISTRY.fromClass(lossFunctionClass));
        metadata.addProperty("updater", UPDATERS_REGISTRY.fromClass(updaterClass));
        
        List<Layer> layers = model.flattened();
        JsonArray array = new JsonArray();
        
        for (int i = 0; i < layers.size(); i++) {
            Layer layer = layers.get(i);
            JsonObject data = new JsonObject();
            
            Class<? extends Layer> layerClass = layer.getClass();
            Class<? extends Activation> activationClass = layer.activation().getClass();
            Class<? extends GradientClipper> clipperClass = layer.clipper().getClass();
            
            Map<String, Tensor> weightsMap = layer.weightsMap();
            String identifier = LAYER_REGISTRY.fromClass(layerClass);
            
            data.addProperty("index", i);
            data.addProperty("type", identifier);
            data.addProperty("activation", ACTIVATION_REGISTRY.fromClass(activationClass));
            data.addProperty("clipper", CLIPPERS_REGISTRY.fromClass(clipperClass));
            
            layer.serialize(data);
            
            JsonArray weightsArray = new JsonArray();
            
            for (Map.Entry<String, Tensor> entry : weightsMap.entrySet()) {
                String id = String.format("%s.%s.%s", identifier, i, entry.getKey());
                globalWeightsMap.put(id, entry.getValue());
                weightsArray.add(id);
            }
            
            data.add("weights", weightsArray);
            array.add(data);
        }
        
        metadata.add("architecture", array);
        
        return GSON.toJson(metadata).getBytes();
    }
    
    private byte[] buildWeights(Map<String, Tensor> globalWeightsMap) {
        JsonObject header = new JsonObject();
        
        int offset = 0;
        
        for (Map.Entry<String, Tensor> entry : globalWeightsMap.entrySet()) {
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
            tensor.add("offset", offsets);
            
            header.add(name, tensor);
        }
        
        byte[] headerJson = GSON.toJson(header).getBytes();
        
        try (ByteArrayOutputStream stream = new ByteArrayOutputStream()) {
            ByteBuffer buffer = ByteBuffer.allocate(8)
                .order(ByteOrder.LITTLE_ENDIAN)
                .putLong(headerJson.length);
            
            stream.write(buffer.array());
            stream.write(headerJson);
            
            for (Tensor tensor : globalWeightsMap.values()) {
                stream.write(tensor.toByteArray());
            }
            
            return stream.toByteArray();
        } catch (IOException e) {
            e.printStackTrace(System.err);
            return new byte[0];
        }
    }
}
