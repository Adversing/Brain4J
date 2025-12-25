package org.brain4j.core.importing;

import org.brain4j.core.graphs.GraphModel;
import org.brain4j.core.importing.impl.BrainFormat;
import org.brain4j.core.importing.impl.OnnxFormat;
import org.brain4j.core.model.Model;

import java.io.File;

public class ModelLoaders {
    
    public static Model fromFile(String path) {
        BrainFormat loader = new BrainFormat();
        File file = new File(path);
        
        return loader.deserialize(file);
    }
    
    public static GraphModel fromOnnx(String path) {
        OnnxFormat loader = new OnnxFormat();
        File file = new File(path);
        
        return loader.deserialize(file);
    }
}
