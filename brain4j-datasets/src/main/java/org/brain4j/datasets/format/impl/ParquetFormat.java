package org.brain4j.datasets.format.impl;

import org.apache.avro.generic.GenericRecord;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.parquet.avro.AvroParquetReader;
import org.apache.parquet.hadoop.ParquetReader;
import org.apache.parquet.hadoop.util.HadoopInputFile;
import org.brain4j.datasets.core.dataset.SampleRecord;
import org.brain4j.datasets.format.FileFormat;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class ParquetFormat implements FileFormat<GenericRecord> {
    @Override
    public String format() {
        return "parquet";
    }
    
    @Override
    public Iterable<GenericRecord> read(File file) throws IOException {
        Path hadoopPath = new Path(file.getPath());
        List<GenericRecord> result = new ArrayList<>();
        
        HadoopInputFile inputFile = HadoopInputFile.fromPath(hadoopPath, new Configuration());
        ParquetReader<GenericRecord> reader = AvroParquetReader.genericRecordReader(inputFile);
        
        GenericRecord record;
        
        while ((record = reader.read()) != null) {
            result.add(record);
        }
        
        reader.close();
        return result;
    }
}
