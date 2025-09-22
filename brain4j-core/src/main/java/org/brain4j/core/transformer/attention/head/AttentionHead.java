package org.brain4j.core.transformer.attention.head;

import org.brain4j.math.Tensors;
import org.brain4j.math.gpu.device.Device;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.index.Range;
import org.brain4j.math.weightsinit.WeightInitialization;
import org.brain4j.math.activation.impl.SoftmaxActivation;
import org.brain4j.math.clipper.GradientClipper;
import org.brain4j.math.data.StatesCache;
import org.brain4j.core.training.optimizer.Optimizer;
import org.brain4j.core.training.updater.Updater;

import java.util.Random;

public class AttentionHead {

    protected final GradientClipper clipper;
    protected Tensor qkvWeights;
    protected int embedDimension;
    protected int headDimension;

    public AttentionHead(GradientClipper clipper, int embedDimension, int headDimension) {
        this.clipper = clipper;
        this.embedDimension = embedDimension;
        this.headDimension = headDimension;
        this.qkvWeights = Tensors.zeros(embedDimension, 3 * headDimension).withGrad();
    }

    public void initWeights(Random generator, WeightInitialization weightInit) {
        qkvWeights.map(_ -> weightInit.generate(generator, embedDimension, 3 * headDimension));
    }

    public void toDevice(Device device) {
        qkvWeights = qkvWeights.to(device);
    }

    public Tensor attend(Tensor input) {
        // input = [batch, seq_length, embedding_dim]
        Tensor QKV = input.matmulGrad(qkvWeights); // [batch, seq_len, 3 * head_dim]
        // [batch, seq_len, head_dim]
        Tensor Q = QKV.sliceGrad(Range.all(), Range.all(), Range.interval(0, headDimension));
        Tensor K = QKV.sliceGrad(Range.all(), Range.all(), Range.interval(headDimension, 2 * headDimension));
        Tensor V = QKV.sliceGrad(Range.all(), Range.all(), Range.interval(2 * headDimension, 3 * headDimension));
        
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

    public void backward(Updater updater, Optimizer optimizer) {
        Tensor qkvGrad = qkvWeights.grad();
        Tensor optimizedQkv = optimizer.step(qkvWeights, qkvGrad);
        
        clipper.clip(optimizedQkv);
        updater.change(qkvWeights, optimizedQkv);
    }

    public void resetGrad() {
        qkvWeights.zerograd();
    }
    
    public GradientClipper clipper() {
        return clipper;
    }
    
    public Tensor qkvWeights() {
        return qkvWeights;
    }
    
    public void setQkvWeights(Tensor qkvWeights) {
        this.qkvWeights = qkvWeights;
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
        return qkvWeights.elements();
    }
}
