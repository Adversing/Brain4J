package org.brain4j.datasets.format;

import org.brain4j.math.Pair;
import org.brain4j.math.tensor.Tensor;

import java.io.IOException;

public interface RecordParser<T> {
    Pair<Tensor[], Tensor[]> parse(T record, int index) throws Exception;
}