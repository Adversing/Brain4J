package org.brain4j.core.importing;

import org.brain4j.core.importing.impl.BrainLoader;
import org.brain4j.core.importing.impl.OnnxLoader;
import org.brain4j.core.model.Model;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;

public class ModelLoaders {
    
    public static Model fromFile(String path) throws Exception {
        byte[] data = Files.readAllBytes(Paths.get(path));
        
        BrainLoader loader = new BrainLoader();
        
        return loader.deserialize(data);
    }
    
    public static Model fromOnnx(String path) throws Exception {
        byte[] data = Files.readAllBytes(Paths.get(path));
        
        OnnxLoader loader = new OnnxLoader();
        
        return loader.deserialize(data);
    }
}
