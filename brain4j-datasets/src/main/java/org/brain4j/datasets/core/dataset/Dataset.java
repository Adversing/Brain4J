package org.brain4j.datasets.core.dataset;

import org.apache.avro.generic.GenericRecord;
import org.apache.hadoop.conf.Configuration;
import org.apache.parquet.avro.AvroParquetReader;
import org.apache.parquet.hadoop.ParquetReader;
import org.apache.parquet.hadoop.util.HadoopInputFile;
import org.brain4j.datasets.api.DatasetInfo;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;

public record Dataset(
        String id,
        DatasetInfo info,
        List<DatasetFile> files,
        Map<String, Object> config
) {

    public record DatasetFile(String name, Path path, long size, String format) {
        
        public List<GenericRecord> extractRecords() throws IOException {
            var hadoopPath = new org.apache.hadoop.fs.Path(path.toString());
            var result = new ArrayList<GenericRecord>();
            
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

    public Optional<DatasetFile> getFile(String filename) {
        return files.stream()
                .filter(file -> file.name().equals(filename))
                .findFirst();
    }

    public List<DatasetFile> getFilesByFormat(String format) {
        return files.stream()
                .filter(file -> file.format().equalsIgnoreCase(format))
                .toList();
    }

    public long getTotalSize() {
        return files.stream()
                .mapToLong(DatasetFile::size)
                .sum();
    }
}