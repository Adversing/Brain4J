package org.brain4j.core.importing;

import org.brain4j.core.graphs.GraphModel;
import org.brain4j.core.importing.impl.BrainFormat;
import org.brain4j.core.importing.impl.OnnxFormat;
import org.brain4j.core.model.Model;

import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.function.Supplier;

public class ModelLoaders {
    
    public static <T extends Model> T fromFile(String path, Supplier<T> constructor) throws Exception {
        byte[] data = Files.readAllBytes(Paths.get(path));
        
        BrainFormat loader = new BrainFormat();
        
        return loader.deserialize(data, constructor);
    }
    
    public static GraphModel fromOnnx(String path) throws Exception {
        byte[] data = Files.readAllBytes(Paths.get(path));
        
        OnnxFormat loader = new OnnxFormat();
        
        return loader.deserialize(data, null);
    }
}
