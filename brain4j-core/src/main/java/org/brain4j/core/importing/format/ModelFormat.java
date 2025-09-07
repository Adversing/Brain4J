package org.brain4j.core.importing.format;

import org.brain4j.core.model.Model;

import java.io.File;
import java.io.IOException;
import java.util.function.Supplier;

public interface ModelFormat {
    <T extends Model> T deserialize(byte[] bytes, Supplier<T> constructor) throws Exception;
    
    void serialize(Model model, File file) throws IOException;
}
