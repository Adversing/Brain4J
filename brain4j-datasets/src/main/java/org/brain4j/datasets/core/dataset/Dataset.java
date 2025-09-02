package org.brain4j.datasets.core.dataset;

import org.brain4j.datasets.api.DatasetInfo;
import org.brain4j.datasets.format.FileFormat;
import org.brain4j.datasets.format.RecordParser;
import org.brain4j.math.Pair;
import org.brain4j.math.data.ListDataSource;
import org.brain4j.math.data.Sample;
import org.brain4j.math.tensor.Tensor;

import java.io.IOException;
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
    
    /**
     * Creates a ListDataSource from a {@link Dataset} object with a custom parser.
     *
     * @param parser a function that parses a sample
     * @param format the format of files to use
     * @param shuffle whether to shuffle the data
     * @param batchSize the size of each batch
     * @return a new ListDataSource containing the loaded data
     * @throws IOException if an error occurs while reading the dataset files
     */
    public <T> ListDataSource createDataSource(
        FileFormat<T> format,
        RecordParser<T> parser,
        boolean shuffle,
        int batchSize
    ) throws IOException {
        List<Sample> samples = new ArrayList<>();
        List<DatasetFile> dataFiles = getFilesByFormat(format.format());
        
        for (DatasetFile file : dataFiles) {
            for (T record : format.read(file.path().toFile())) {
                Pair<Tensor[], Tensor[]> pair = parser.parse(record, samples.size());
                
                if (pair == null) continue;
                
                samples.add(new Sample(pair.first(), pair.second()));
            }
        }
        
        return new ListDataSource(samples, shuffle, batchSize);
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