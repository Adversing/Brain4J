package org.brain4j.datasets.format.impl;

import org.apache.hadoop.fs.Path;
import org.apache.parquet.example.data.Group;
import org.apache.parquet.hadoop.ParquetReader;
import org.apache.parquet.hadoop.example.GroupReadSupport;
import org.brain4j.datasets.format.FileFormat;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class ParquetFormat implements FileFormat<Group> {

    @Override
    public String format() {
        return "parquet";
    }
    
    @Override
    public Iterable<Group> read(File file) throws IOException {
        Path hadoopPath = new Path(file.getPath());
        List<Group> result = new ArrayList<>();
        
        try (ParquetReader<Group> reader = ParquetReader.builder(new GroupReadSupport(), hadoopPath).build()) {
            Group group;
            
            while ((group = reader.read()) != null) {
                result.add(group);
            }
        }
        
        return result;
    }
}
