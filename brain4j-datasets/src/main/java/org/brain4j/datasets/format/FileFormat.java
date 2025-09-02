package org.brain4j.datasets.format;

import java.io.File;
import java.io.IOException;

public interface FileFormat<T> {
    String format();
    
    Iterable<T> read(File file) throws IOException;
}
