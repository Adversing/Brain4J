package org.brain4j.datasets.format;

import org.brain4j.math.commons.Pair;
import org.brain4j.math.tensor.Tensor;

public interface RecordParser<T> {
    Pair<Tensor[], Tensor[]> parse(T record, int index) throws Exception;
}