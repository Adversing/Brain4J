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