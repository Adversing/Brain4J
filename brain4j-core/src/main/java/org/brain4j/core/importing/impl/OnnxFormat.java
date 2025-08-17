package org.brain4j.core.importing.impl;

import org.brain4j.math.Tensors;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.autograd.Operation;
import org.brain4j.core.graphs.GraphModel;
import org.brain4j.core.graphs.GraphNode;
import org.brain4j.core.importing.ModelFormat;
import org.brain4j.core.importing.onnx.ProtoOnnx;
import org.brain4j.core.model.Model;

import java.io.File;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.Supplier;

@SuppressWarnings("unchecked")
public class OnnxFormat implements ModelFormat {
    
    @Override
    public <T extends Model> T deserialize(byte[] bytes, Supplier<T> constructor) throws Exception {
        ProtoOnnx.ModelProto proto = ProtoOnnx.ModelProto.parseFrom(bytes);
        ProtoOnnx.GraphProto graph = proto.getGraph();
        
        GraphModel.Builder model = GraphModel.newGraph();
        
        for (ProtoOnnx.TensorProto tensor : graph.getInitializerList()) {
            Tensor weight = deserializeTensor(tensor);
            model.initializer(tensor.getName(), weight);
        }
        
        for (ProtoOnnx.NodeProto node : graph.getNodeList()) {
            Operation operation = OPERATION_MAP.get(node.getOpType());
            
            if (operation == null) {
                throw new IllegalArgumentException("Unknown or missing operation type: " + node.getOpType());
            }
            
            if (node.getInputCount() != operation.requiredInputs()) {
                throw new IllegalArgumentException(
                    "Node " + node.getOpType() + " requires " + node.getInputCount()
                        + " inputs but operation requires " + operation.requiredInputs()
                );
            }
            
            model.addNode(new GraphNode(
                node.getName(),
                operation,
                node.getInputList(),
                node.getOutputList()
            ));
        }
        
        List<String> inputs = graph.getInputList().stream()
            .map(ProtoOnnx.ValueInfoProto::getName)
            .toList();
        
        List<String> outputs = graph.getOutputList().stream()
            .map(ProtoOnnx.ValueInfoProto::getName)
            .toList();

        return (T) model.inputs(inputs)
                .outputs(outputs)
                .compile();
    }
    
    @Override
    public void serialize(Model model, File file) {
        throw new UnsupportedOperationException();
    }

    private ProtoOnnx.TensorProto serializeTensor(Tensor tensor) {
        ProtoOnnx.TensorProto.Builder builder = ProtoOnnx.TensorProto.newBuilder();

        List<Long> dimensions = Arrays
            .stream(tensor.shape())
            .asLongStream()
            .boxed()
            .toList();

        List<Float> data = new ArrayList<>();

        for (float value : tensor.data()) {
            data.add(value);
        }

        return builder
            .addAllDims(dimensions)
            .addAllFloatData(data)
            .build();
    }

    private Tensor deserializeTensor(ProtoOnnx.TensorProto tensor) {
        byte[] rawData = tensor.getRawData().toByteArray();

        ByteBuffer dataBuffer = ByteBuffer.wrap(rawData).order(ByteOrder.LITTLE_ENDIAN);
        List<Long> dimensions = tensor.getDimsList();

        int[] shape = new int[dimensions.size()];

        for (int i = 0; i < shape.length; i++) {
            shape[i] = Math.toIntExact(dimensions.get(i));
        }

        Tensor result = Tensors.create(shape);
        float[] data = result.data();
        
        for (int i = 0; i < data.length; i++) {
            data[i] = dataBuffer.getFloat();
        }
        
        return result;
    }
}
