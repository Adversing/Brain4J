package org.brain4j.core.transformer.attention.head;

import org.brain4j.common.Tensors;
import org.brain4j.common.gpu.device.Device;
import org.brain4j.common.tensor.Tensor;
import org.brain4j.common.weightsinit.WeightInitialization;
import org.brain4j.core.activation.impl.SoftmaxActivation;
import org.brain4j.core.clipper.GradientClipper;
import org.brain4j.core.importing.proto.ProtoModel;
import org.brain4j.core.importing.proto.SerializeUtils;
import org.brain4j.core.training.StatesCache;
import org.brain4j.core.training.optimizer.Optimizer;
import org.brain4j.core.training.updater.Updater;

import java.util.Random;

public class AttentionHead {

    protected final GradientClipper clipper;
    
    protected Tensor queryWeights;
    protected Tensor keyWeights;
    protected Tensor valueWeights;

    protected int embedDimension;
    protected int headDimension;

    public AttentionHead(GradientClipper clipper, int embedDimension, int headDimension) {
        this.clipper = clipper;
        this.embedDimension = embedDimension;
        this.headDimension = headDimension;

        this.queryWeights = Tensors.zeros(embedDimension, headDimension).withGrad();
        this.keyWeights = Tensors.zeros(embedDimension, headDimension).withGrad();
        this.valueWeights = Tensors.zeros(embedDimension, headDimension).withGrad();
    }

    public void initWeights(Random generator, WeightInitialization weightInit) {
        this.queryWeights.map(x -> weightInit.generate(generator, embedDimension, headDimension));
        this.keyWeights.map(x -> weightInit.generate(generator, embedDimension, headDimension));
        this.valueWeights.map(x -> weightInit.generate(generator, embedDimension, headDimension));
    }

    public void to(Device device) {
        this.queryWeights.to(device);
        this.keyWeights.to(device);
        this.valueWeights.to(device);
    }

    public Tensor attend(Tensor input) {
        // input = [batch_size, seq_length, embedding_dim]
        Tensor Q = input.matmulGrad(queryWeights); // [batch_size, seq_length, head_dimension]
        Tensor K = input.matmulGrad(keyWeights); // [batch_size, seq_length, head_dimension]
        Tensor V = input.matmulGrad(valueWeights); // [batch_size, seq_length, head_dimension]
        
        double normalizer = Math.sqrt(headDimension);
        
        // [batch_size, head_dimension, seq_length]
        Tensor K_T = K.transposeGrad();
        // [batch_size, seq_length, seq_length]
        Tensor scores = Q.matmulGrad(K_T).div(normalizer);
        Tensor attentionWeights = scores.activateGrad(new SoftmaxActivation());

        // [batch_size, seq_length, head_dimension]
        return attentionWeights.matmulGrad(V);
    }

    public Tensor attend(StatesCache cache, Tensor input) {
        return attend(input);
    }

    public Tensor queryWeights() {
        return queryWeights;
    }

    public Tensor keyWeights() {
        return keyWeights;
    }

    public Tensor valueWeights() {
        return valueWeights;
    }

    public int embedDimension() {
        return embedDimension;
    }

    public int headDimension() {
        return headDimension;
    }

    public int totalWeights() {
        return queryWeights.elements() + keyWeights.elements() + valueWeights.elements();
    }

    public void backward(Updater updater, Optimizer optimizer) {
        Tensor queryGrad = queryWeights.grad();
        Tensor keyGrad = keyWeights.grad();
        Tensor valueGrad = valueWeights.grad();

        Tensor optimizedQuery = optimizer.step(queryWeights, queryGrad);
        Tensor optimizedKey = optimizer.step(keyWeights, keyGrad);
        Tensor optimizedValue = optimizer.step(valueWeights, valueGrad);
        
        clipper.clip(optimizedQuery);
        clipper.clip(optimizedKey);
        clipper.clip(optimizedValue);
        
        updater.change(queryWeights, optimizedQuery);
        updater.change(keyWeights, optimizedKey);
        updater.change(valueWeights, optimizedValue);
    }
    
    public ProtoModel.AttentionHead serialize() {
        return ProtoModel.AttentionHead.newBuilder()
            .setKWeight(SerializeUtils.serializeTensor("key_weight", keyWeights))
            .setQWeight(SerializeUtils.serializeTensor("query_weight", queryWeights))
            .setVWeight(SerializeUtils.serializeTensor("value_weight", valueWeights))
            .build();
    }
    
    public void deserialize(ProtoModel.AttentionHead protoHead) {
        this.keyWeights = SerializeUtils.deserializeTensor(protoHead.getKWeight());
        this.queryWeights = SerializeUtils.deserializeTensor(protoHead.getQWeight());
        this.valueWeights = SerializeUtils.deserializeTensor(protoHead.getVWeight());
    }
    
    public void resetGrad() {
        keyWeights.zerograd();
        queryWeights.zerograd();
        valueWeights.zerograd();
    }
}
