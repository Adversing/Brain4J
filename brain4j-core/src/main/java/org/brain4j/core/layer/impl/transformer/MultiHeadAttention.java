package org.brain4j.core.layer.impl.transformer;

import org.brain4j.core.layer.Layer;
import org.brain4j.core.training.optimizer.Optimizer;
import org.brain4j.core.training.updater.Updater;
import org.brain4j.core.transformer.attention.MaskedMultiHeadAttention;
import org.brain4j.core.transformer.attention.head.AttentionHead;
import org.brain4j.math.Tensors;
import org.brain4j.math.activation.impl.SoftmaxActivation;
import org.brain4j.math.clipper.GradientClipper;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.gpu.device.Device;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.index.Range;

import java.util.ArrayList;
import java.util.List;
import java.util.random.RandomGenerator;

/**
 * Implements the Multi-Head Attention mechanism as used in Transformer architectures.
 * <p>
 * This class manages the projection of input embeddings into queries (Q), keys (K), and values (V),
 * distributes them across multiple attention heads, computes attention scores using scaled dot-product attention,
 * and combines the outputs into a single representation. It also supports gradient clipping, weight initialization,
 * device placement, and optimization via {@link Optimizer} and {@link Updater}.
 * </p>
 * <h2>Shape conventions:</h2>
 * <ul>
 *   <li>Input: {@code [batch, seq_len, embedding_dim]}</li>
 *   <li>Q, K, V: {@code [batch, heads, seq_len, head_dim]}</li>
 *   <li>Attention scores: {@code [batch, heads, seq_len, seq_len]}</li>
 *   <li>Output: {@code [batch, seq_len, embedding_dim]}</li>
 * </ul>
 *
 * @see AttentionHead
 * @see Tensor
 * @see Optimizer
 * @see Updater
 * @author xEcho1337
 */
public class MultiHeadAttention extends Layer {
    
    protected Tensor keyProj;
    protected Tensor queryProj;
    protected Tensor valueProj;
    protected Tensor outProj;
    protected Tensor outBias;
    protected int headCount;
    protected int embeddingDim;
    protected int headDimension;
    protected boolean attnQkvHasBias;
    protected boolean attnOutHasBias;

    private MultiHeadAttention() {
    }

    /**
     * Configures the Multi-Head attention mechanism with the specified parameters.
     * @param clipper the gradient clipper to use during training
     * @param headCount the amount of heads in the MHA; MUST be a multiple of the embedding dimension
     * @param embeddingDim the embedding dimension of the transformer
     */
    public MultiHeadAttention(GradientClipper clipper, int headCount, int embeddingDim) {
        this.clipper = clipper;
        this.headCount = headCount;
        this.embeddingDim = embeddingDim;
        this.attnQkvHasBias = true;

        if (embeddingDim % headCount != 0) {
            throw new IllegalArgumentException("Embedding dimension must be divisible by head count! (%s %% %s = %s)"
                    .formatted(embeddingDim, headCount, embeddingDim % headCount));
        }

        this.headDimension = embeddingDim / headCount;
    }

    @Override
    public Layer connect(Layer previous) {
        this.keyProj = Tensors.zeros(embeddingDim, embeddingDim).withGrad();
        this.queryProj = Tensors.zeros(embeddingDim, embeddingDim).withGrad();
        this.valueProj = Tensors.zeros(embeddingDim, embeddingDim).withGrad();
        this.outProj = Tensors.matrix(embeddingDim, embeddingDim).withGrad();

        if (attnQkvHasBias) this.bias = Tensors.zeros(3 * embeddingDim).withGrad();
        if (attnOutHasBias) this.outBias = Tensors.zeros(embeddingDim).withGrad();

        return this;
    }

    @Override
    public void initWeights(RandomGenerator generator, int input, int output) {
        this.keyProj.map(x -> weightInit.generate(generator, embeddingDim, embeddingDim));
        this.queryProj.map(x -> weightInit.generate(generator, embeddingDim, embeddingDim));
        this.valueProj.map(x -> weightInit.generate(generator, embeddingDim, embeddingDim));
        this.outProj.map(x -> weightInit.generate(generator, embeddingDim, embeddingDim));
        this.weights = Tensors.concat(keyProj, queryProj, valueProj).withGrad();
    }

    /**
     * Computes the forward pass of the multi-head attention mechanism for a given input tensor.
     * <p>This implementation follows the original architecture defined
     * by the <a href="https://arxiv.org/abs/1706.03762">original paper</a>.
     * @param cache cache used to store intermediate results
     * @param inputs the input tensors, first tensor must have shape {@code [batch, seq_len, embedding_dim]}
     * @return the output tensor of shape {@code [batch, seq_len, embedding_dim]}
     */
    @Override
    public Tensor[] forward(StatesCache cache, Tensor... inputs) {
        Tensor input = inputs[0];
        int batch = input.shape(0);
        int seqLength = input.shape(1);

        // [batch, seq_len, 3 * H * head_dim]
        Tensor QKV = input.matmulGrad(weights);

        if (attnQkvHasBias) QKV = QKV.addGrad(bias);

        Tensor reshaped = QKV.reshapeGrad(batch, seqLength, headCount, 3, headDimension);

        // [batch, heads, seq_len, 3, head_dim]
        reshaped = reshaped.transposeGrad(1, 2);

        Tensor[] QKVs = new Tensor[3];
        Range all = Range.all();

        for (int i = 0; i < QKVs.length; i++) {
            // [batch, heads, seq_len, 1, head_dim]
            QKVs[i] = reshaped.sliceGrad(all, all, all, Range.point(i), all);
            // [batch, heads, seq_len, head_dim]
            QKVs[i] = QKVs[i].squeezeGrad(3);
        }

        // [batch, heads, seq_len, head_dim]
        Tensor Q = QKVs[0], K = QKVs[1], V = QKVs[2];

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
        // [batch, seq_len, embedding_dim]
        Tensor output = context.reshapeGrad(batch, seqLength, embeddingDim);
        // [batch, seq_len, embedding_dim]
        
        Tensor result = output.matmulGrad(outProj);
        
        if (attnOutHasBias) result = result.addGrad(outBias);
        
        return new Tensor[] { result };
    }

    @Override
    public void backward(StatesCache cache, Updater updater, Optimizer optimizer) {
        super.backward(cache, updater, optimizer);

        Tensor optimized = optimizer.step(outProj, outProj.grad());
        clipper.clip(optimized);
        updater.change(outProj, optimized);
        
        if (attnOutHasBias) {
            Tensor biasGrad = outBias.grad().sum(0, false);
            clipper.clip(biasGrad);
            updater.change(outBias, biasGrad);
        }
    }

    @Override
    public void toDevice(Device device) {
        this.weights = weights.to(device);
        this.outProj = outProj.to(device);
        if (attnQkvHasBias) this.bias = bias.to(device);
        if (attnOutHasBias) this.outBias = outBias.to(device);
    }

    @Override
    public int size() {
        return embeddingDim;
    }

    @Override
    public int totalWeights() {
        return weights.elements() + outProj.elements();
    }
    
    /**
     * Resets the autograd state.
     */
    public void resetGrad() {
        super.resetGrad();
        outProj.zerograd();
        if (attnOutHasBias) outBias.zerograd();
    }
    
    public Tensor keyProj() {
        return keyProj;
    }
    
    public MultiHeadAttention setKeyProj(Tensor keyProj) {
        this.keyProj = keyProj;
        return this;
    }
    
    public Tensor queryProj() {
        return queryProj;
    }
    
    public MultiHeadAttention setQueryProj(Tensor queryProj) {
        this.queryProj = queryProj;
        return this;
    }
    
    public Tensor valueProj() {
        return valueProj;
    }
    
    public MultiHeadAttention setValueProj(Tensor valueProj) {
        this.valueProj = valueProj;
        return this;
    }
    
    public Tensor outProj() {
        return outProj;
    }
    
    public MultiHeadAttention setOutProj(Tensor outProj) {
        this.outProj = outProj;
        return this;
    }
    
    public Tensor outBias() {
        return outBias;
    }
    
    public MultiHeadAttention setOutBias(Tensor outBias) {
        this.outBias = outBias;
        return this;
    }
    
    public int headCount() {
        return headCount;
    }
    
    public MultiHeadAttention setHeadCount(int headCount) {
        this.headCount = headCount;
        return this;
    }
    
    public int embeddingDim() {
        return embeddingDim;
    }
    
    public MultiHeadAttention setEmbeddingDim(int embeddingDim) {
        this.embeddingDim = embeddingDim;
        return this;
    }
    
    public int headDimension() {
        return headDimension;
    }
    
    public MultiHeadAttention setHeadDimension(int headDimension) {
        this.headDimension = headDimension;
        return this;
    }
    
    public boolean attnQkvHasBias() {
        return attnQkvHasBias;
    }
    
    public MultiHeadAttention setAttnQkvHasBias(boolean attnQkvHasBias) {
        this.attnQkvHasBias = attnQkvHasBias;
        return this;
    }
    
    public boolean attnOutHasBias() {
        return attnOutHasBias;
    }
    
    public MultiHeadAttention setAttnOutHasBias(boolean attnOutHasBias) {
        this.attnOutHasBias = attnOutHasBias;
        return this;
    }
}
