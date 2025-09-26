package org.brain4j.core.transformer.attention;

import org.brain4j.math.Tensors;
import org.brain4j.math.activation.impl.SoftmaxActivation;
import org.brain4j.math.gpu.device.Device;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.index.Range;
import org.brain4j.math.weightsinit.WeightInitialization;
import org.brain4j.math.clipper.GradientClipper;
import org.brain4j.math.data.StatesCache;
import org.brain4j.core.training.optimizer.Optimizer;
import org.brain4j.core.training.updater.Updater;
import org.brain4j.core.transformer.attention.head.AttentionHead;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class MultiHeadAttention {

    protected GradientClipper clipper;
    protected List<AttentionHead> heads;
    protected Tensor outProjWeights;
    protected Tensor qkvWeights;
    protected int headCount;
    protected int embeddingDim;
    protected int headDimension;
    
    public MultiHeadAttention() {
        this.heads = new ArrayList<>();
    }
    
    public MultiHeadAttention(GradientClipper clipper, int headCount, int embeddingDim) {
        this.clipper = clipper;
        this.headCount = headCount;
        this.embeddingDim = embeddingDim;

        if (embeddingDim % headCount != 0) {
            throw new IllegalArgumentException("Embedding dimension must be divisible by head count! (%s %% %s = %s)"
                    .formatted(embeddingDim, headCount, embeddingDim % headCount));
        }

        this.headDimension = embeddingDim / headCount;
        this.heads = new ArrayList<>();
        this.outProjWeights = Tensors.matrix(embeddingDim, embeddingDim).withGrad();
        this.qkvWeights = Tensors.matrix(embeddingDim, 3 * headCount * headDimension).withGrad();

        initializeHeads();
    }
    
    public AttentionHead createAttentionHead() {
        return new AttentionHead(clipper, embeddingDim, headDimension);
    }

    public void toDevice(Device device) {
        for (AttentionHead head : heads) {
            head.toDevice(device);
        }

        outProjWeights = outProjWeights.to(device);
        qkvWeights = qkvWeights.to(device);
    }

    public void initWeights(Random generator, WeightInitialization weightInit) {
        for (AttentionHead head : heads) {
            head.initWeights(generator, weightInit);
        }

        this.outProjWeights.map(_ -> weightInit.generate(generator, embeddingDim, embeddingDim));
        this.qkvWeights.map(_ -> weightInit.generate(generator, embeddingDim, embeddingDim));
    }
    
    protected void initializeHeads() {
        for (int i = 0; i < headCount; i++) {
            heads.add(createAttentionHead());
        }
    }

    public Tensor attend(StatesCache cache, Tensor input) {
        int batch = input.shape(0);
        int seqLength = input.shape(1);

        // [batch, seq_len, 3 * H * head_dim]
        Tensor QKV = input.matmulGrad(qkvWeights);
        Tensor reshaped = QKV.reshapeGrad(batch, seqLength, headCount, 3, headDimension)
            // [batch, heads, seq_len, 3, head_dim]
            .transposeGrad(1, 2);

        Tensor[] QKVs = new Tensor[3];
        Range all = Range.all();

        for (int i = 0; i < QKVs.length; i++) {
            // [batch, heads, seq_len, head_dim]
            QKVs[i] = reshaped.sliceGrad(all, all, all, Range.point(i), all).squeezeGrad(3);
        }

        // [batch, heads, seq_len, head_dim]
        Tensor Q = QKVs[0];
        Tensor K = QKVs[1];
        Tensor V = QKVs[2];

        double normalizer = Math.sqrt(headDimension);

        // [batch, heads, head_dim, seq_len]
        Tensor K_T = K.transposeGrad();
        // [batch, heads, seq_len, seq_len]
        Tensor scores = Q.matmulGrad(K_T).div(normalizer);
        Tensor attentionWeights = scores.activateGrad(new SoftmaxActivation());

        // [batch, heads, seq_len, head_dim]
        Tensor context = attentionWeights.matmulGrad(V);

        // [batch, seq_len, heads, head_dim]
        context = context.transposeGrad(1, 2);

        // [batch, seq_len, heads * head_dim]
        Tensor output = context.reshapeGrad(batch, seqLength, headCount * headDimension);

        return output.matmulGrad(outProjWeights);
    }

    public int totalWeights() {
        return 3 * embeddingDim * embeddingDim + outProjWeights.elements();
    }

    public List<AttentionHead> heads() {
        return heads;
    }

    public void backward(Updater updater, Optimizer optimizer) {
        Tensor outWeightsGrad = optimizer.step(outProjWeights, outProjWeights.grad());
        Tensor qkvGrad = optimizer.step(qkvWeights, qkvWeights.grad());

        clipper.clip(outWeightsGrad);
        clipper.clip(qkvGrad);

        updater.change(outProjWeights, outWeightsGrad);
        updater.change(qkvWeights, qkvGrad);
    }

    public void resetGrad() {
        for (AttentionHead head : heads()) {
            head.resetGrad();
        }

        qkvWeights.zerograd();
        outProjWeights.zerograd();
    }

    public GradientClipper clipper() {
        return clipper;
    }
    
    public void setClipper(GradientClipper clipper) {
        this.clipper = clipper;
    }
    
    public Tensor outProjWeights() {
        return outProjWeights;
    }
    
    public void setOutProjWeights(Tensor outProjWeights) {
        this.outProjWeights = outProjWeights;
    }
    
    public int headCount() {
        return headCount;
    }
    
    public void setHeadCount(int headCount) {
        this.headCount = headCount;
    }
    
    public int embeddingDim() {
        return embeddingDim;
    }
    
    public void setEmbeddingDim(int embeddingDim) {
        this.embeddingDim = embeddingDim;
    }
    
    public int headDimension() {
        return headDimension;
    }
    
    public void setHeadDimension(int headDimension) {
        this.headDimension = headDimension;
    }
}
