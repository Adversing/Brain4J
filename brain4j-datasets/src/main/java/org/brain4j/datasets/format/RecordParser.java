package org.brain4j.datasets.format;

import org.brain4j.math.commons.Batch;

public interface RecordParser<T> {
    Batch parse(T record, int index) throws Exception;
}