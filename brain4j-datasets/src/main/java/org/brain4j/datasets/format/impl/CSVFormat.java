package org.brain4j.datasets.format.impl;

import org.brain4j.datasets.format.FileFormat;

import java.io.File;
import java.io.IOException;

public class CSVFormat implements FileFormat<String> {
    @Override
    public String format() {
        return "csv";
    }
    
    @Override
    public Iterable<String> read(File file) throws IOException {
        throw new UnsupportedOperationException();
    }
}
