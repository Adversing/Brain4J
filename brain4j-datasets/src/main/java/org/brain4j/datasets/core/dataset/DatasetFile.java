package org.brain4j.datasets.core.dataset;

import org.apache.avro.generic.GenericRecord;
import org.apache.hadoop.conf.Configuration;
import org.apache.parquet.avro.AvroParquetReader;
import org.apache.parquet.hadoop.ParquetReader;
import org.apache.parquet.hadoop.util.HadoopInputFile;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

public record DatasetFile(String name, Path path, long size, String format) {
    public List<SampleRecord> extractRecords() throws IOException {
        var hadoopPath = new org.apache.hadoop.fs.Path(path.toString());
        var result = new ArrayList<SampleRecord>();
        
        HadoopInputFile inputFile = HadoopInputFile.fromPath(hadoopPath, new Configuration());
        ParquetReader<GenericRecord> reader = AvroParquetReader.genericRecordReader(inputFile);
        
        GenericRecord record;
        
        while ((record = reader.read()) != null) {
            result.add(new SampleRecord(record));
        }
        
        reader.close();
        return result;
    }
}