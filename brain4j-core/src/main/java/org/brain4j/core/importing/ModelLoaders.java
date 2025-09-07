package org.brain4j.core.importing;

import org.brain4j.core.graphs.GraphModel;
import org.brain4j.core.importing.impl.BrainFormat;
import org.brain4j.core.importing.impl.OnnxFormat;
import org.brain4j.core.model.Model;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.function.Supplier;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

public class ModelLoaders {
    
    public static <T extends Model> T fromFile(String path, Supplier<T> constructor) {
        BrainFormat loader = new BrainFormat();
        File file = new File(path);
        
        return loader.deserialize(file, constructor);
    }
    
    public static GraphModel fromOnnx(String path) {
        OnnxFormat loader = new OnnxFormat();
        File file = new File(path);
        
        return loader.deserialize(file, null);
    }
}
