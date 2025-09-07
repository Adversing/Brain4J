package org.brain4j.core.transformer.attention.head;

import org.brain4j.math.Tensors;
import org.brain4j.math.gpu.device.Device;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.weightsinit.WeightInitialization;
import org.brain4j.core.activation.impl.SoftmaxActivation;
import org.brain4j.core.clipper.GradientClipper;
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
        queryWeights.map(x -> weightInit.generate(generator, embedDimension, headDimension));
        keyWeights.map(x -> weightInit.generate(generator, embedDimension, headDimension));
        valueWeights.map(x -> weightInit.generate(generator, embedDimension, headDimension));
    }

    public void toDevice(Device device) {
        queryWeights.to(device);
        keyWeights.to(device);
        valueWeights.to(device);
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
    
    public GradientClipper clipper() {
        return clipper;
    }
    
    public Tensor queryWeights() {
        return queryWeights;
    }
    
    public void setQueryWeights(Tensor queryWeights) {
        this.queryWeights = queryWeights;
    }
    
    public Tensor keyWeights() {
        return keyWeights;
    }
    
    public void setKeyWeights(Tensor keyWeights) {
        this.keyWeights = keyWeights;
    }
    
    public Tensor valueWeights() {
        return valueWeights;
    }
    
    public void setValueWeights(Tensor valueWeights) {
        this.valueWeights = valueWeights;
    }
    
    public void setEmbedDimension(int embedDimension) {
        this.embedDimension = embedDimension;
    }
    
    public void setHeadDimension(int headDimension) {
        this.headDimension = headDimension;
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
    
    public void resetGrad() {
        keyWeights.zerograd();
        queryWeights.zerograd();
        valueWeights.zerograd();
    }
}
