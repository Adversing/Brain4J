package org.brain4j.core.graphs;

import org.brain4j.math.tensor.autograd.Operation;

import java.util.List;

public record GraphNode(String name, Operation operation, List<String> inputs, List<String> outputs) {
}