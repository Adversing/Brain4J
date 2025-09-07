package org.brain4j.core.importing.impl;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import org.brain4j.core.clipper.GradientClipper;
import org.brain4j.core.layer.Layer;
import org.brain4j.core.importing.format.ModelFormat;
import org.brain4j.core.model.Model;
import org.brain4j.math.activation.Activation;
import org.brain4j.math.tensor.Tensor;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.time.Instant;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;
import java.util.zip.Deflater;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;

import static org.brain4j.core.importing.Registries.*;

public class BrainFormat implements ModelFormat {
    
    public static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();
    public static final int FORMAT_VERSION = 1;

    public BrainFormat() {
    }

    @Override
    public <T extends Model> T deserialize(byte[] bytes, Supplier<T> constructor) throws Exception {
        return constructor.get();
    }
    
    @Override
    public void serialize(Model model, File file) {
        Map<String, Tensor> globalWeightsMap = new HashMap<>();
        
        String metadata = buildMetadata(model, globalWeightsMap);
        String weights = buildWeights(model, globalWeightsMap);
        
        String[] files = { "metadata.json", "weights.safetensors" };
        String[] content = { metadata, weights };
        
        try (ByteArrayOutputStream baos = new ByteArrayOutputStream();
             ZipOutputStream zos = new ZipOutputStream(baos)) {
            zos.setLevel(Deflater.BEST_COMPRESSION);
            
            for (int i = 0; i < files.length; i++) {
                ZipEntry metaEntry = new ZipEntry(files[i]);
                byte[] fileContent = content[i].getBytes();
                
                metaEntry.setMethod(ZipEntry.DEFLATED);
                metaEntry.setSize(fileContent.length);
                
                zos.putNextEntry(metaEntry);
                zos.write(fileContent);
            }
            
            zos.closeEntry();
            zos.close();
            
            byte[] zipData = baos.toByteArray();
            
            try (FileOutputStream fos = new FileOutputStream(file)) {
                fos.write(zipData);
            }
        } catch (IOException e) {
            e.printStackTrace(System.err);
        }
    }
    
    private String buildWeights(Model model, Map<String, Tensor> globalWeightsMap) {
        return "SKIBIDI\nWOO";
    }
    
    private String buildMetadata(Model model, Map<String, Tensor> globalWeightsMap) {
        JsonObject object = new JsonObject();
        Instant date = Instant.now();
        
        object.addProperty("format_version", FORMAT_VERSION);
        object.addProperty("created_at", date.toString());
        object.addProperty("weights_file", "weights.safetensors");
        
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
        
        object.add("architecture", array);
        
        return GSON.toJson(object);
    }
}
