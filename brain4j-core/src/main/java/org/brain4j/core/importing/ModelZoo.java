package org.brain4j.core.importing;

import org.brain4j.core.graphs.GraphModel;
import org.brain4j.core.importing.format.impl.BrainFormat;
import org.brain4j.core.importing.format.impl.OnnxFormat;
import org.brain4j.core.model.Model;

import java.io.File;

public class ModelZoo {
    
    public static final BrainFormat BRAIN_FORMAT = new BrainFormat();
    public static final OnnxFormat ONNX_FORMAT = new OnnxFormat();
    
    public static void saveModel(Model model, File file) {
        BRAIN_FORMAT.serialize(model, file);
    }
    
    public static void saveOnnx(Model model, File file) {
        ONNX_FORMAT.serialize(model, file);
    }
    
    public static Model fromFile(String path) {
        return BRAIN_FORMAT.deserialize(new File(path));
    }
    
    public static GraphModel fromOnnx(String path) {
        return ONNX_FORMAT.deserialize(new File(path));
    }
}
