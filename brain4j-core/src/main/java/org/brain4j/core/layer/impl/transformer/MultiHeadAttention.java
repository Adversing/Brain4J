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

    protected List<AttentionHead> heads; // TODO: migrate to single tensor
    protected Tensor keyProj;
    protected Tensor queryProj;
    protected Tensor valueProj;
    protected Tensor outProj;
    protected int headCount;
    protected int embeddingDim;
    protected int headDimension;
    protected boolean hasBias;

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
        this.hasBias = true;

        if (embeddingDim % headCount != 0) {
            throw new IllegalArgumentException("Embedding dimension must be divisible by head count! (%s %% %s = %s)"
                    .formatted(embeddingDim, headCount, embeddingDim % headCount));
        }

        this.headDimension = embeddingDim / headCount;
        this.heads = new ArrayList<>();
    }
    
    /**
     * Creates a new attention head based on the type of attention mechanism.
     * <p>This method will return a new {@link AttentionHead} instance if called by
     * the default {@link MultiHeadAttention}. If the MHA instance is a {@link MaskedMultiHeadAttention},
     * this method will return a new {@link org.brain4j.core.transformer.attention.head.MaskedAttentionHead}.
     * @deprecated this method will be removed in future releases
     * @return a new attention head instance
     */
    @Deprecated(forRemoval = true)
    public AttentionHead createAttentionHead() {
        return new AttentionHead(clipper, embeddingDim, headDimension);
    }

    @Override
    public Layer connect(Layer previous) {
        this.keyProj = Tensors.zeros(embeddingDim, embeddingDim).withGrad();
        this.queryProj = Tensors.zeros(embeddingDim, embeddingDim).withGrad();
        this.valueProj = Tensors.zeros(embeddingDim, embeddingDim).withGrad();
        this.outProj = Tensors.matrix(embeddingDim, embeddingDim).withGrad();

        if (hasBias) {
            this.bias = Tensors.zeros(3 * embeddingDim).withGrad();
        }

        return this;
    }

    @Override
    public void initWeights(RandomGenerator generator, int input, int output) {
        for (AttentionHead head : heads) {
            head.initWeights(generator, weightInit);
        }

        this.keyProj.map(x -> weightInit.generate(generator, embeddingDim, embeddingDim));
        this.queryProj.map(x -> weightInit.generate(generator, embeddingDim, embeddingDim));
        this.valueProj.map(x -> weightInit.generate(generator, embeddingDim, embeddingDim));
        this.outProj.map(x -> weightInit.generate(generator, embeddingDim, embeddingDim));
        this.weights.map(x -> weightInit.generate(generator, embeddingDim, embeddingDim));
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

        if (hasBias) {
            QKV = QKV.addGrad(bias);
        }

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
        return new Tensor[] { output.matmulGrad(outProj) };
    }

    @Override
    public void backward(StatesCache cache, Updater updater, Optimizer optimizer) {
        super.backward(cache, updater, optimizer);

        Tensor optimized = optimizer.step(outProj, outProj.grad());
        clipper.clip(optimized);
        updater.change(outProj, optimized);
    }

    @Override
    public void toDevice(Device device) {
        for (AttentionHead head : heads) {
            head.toDevice(device);
        }

        this.outProj = outProj.to(device);
        this.weights = weights.to(device);
        this.bias = bias.to(device);
    }

    @Override
    public int size() {
        return embeddingDim;
    }

    @Deprecated(forRemoval = true)
    protected void initializeHeads() {
        for (int i = 0; i < headCount; i++) {
            heads.add(createAttentionHead());
        }
    }

    @Override
    public int totalWeights() {
        return weights.elements() + outProj.elements();
    }

    /**
     * Returns the list of attention heads managed by this layer.
     * @return a list of {@link AttentionHead} objects
     */
    @Deprecated(forRemoval = true)
    public List<AttentionHead> heads() {
        return heads;
    }
    
    /**
     * Resets the autograd state.
     */
    public void resetGrad() {
        super.resetGrad();

        for (AttentionHead head : heads()) {
            head.resetGrad();
        }

        outProj.zerograd();
    }

    public MultiHeadAttention setHeads(List<AttentionHead> heads) {
        this.heads = heads;
        return this;
    }

    public Tensor outProj() {
        return outProj;
    }

    public MultiHeadAttention outProj(Tensor outProj) {
        this.outProj = outProj;
        return this;
    }

    public int headCount() {
        return headCount;
    }

    public MultiHeadAttention headCount(int headCount) {
        this.headCount = headCount;
        return this;
    }

    public int embeddingDim() {
        return embeddingDim;
    }

    public MultiHeadAttention embeddingDim(int embeddingDim) {
        this.embeddingDim = embeddingDim;
        return this;
    }

    public int headDimension() {
        return headDimension;
    }

    public MultiHeadAttention headDimension(int headDimension) {
        this.headDimension = headDimension;
        return this;
    }

    public boolean hasBias() {
        return hasBias;
    }

    public MultiHeadAttention hasBias(boolean hasBias) {
        this.hasBias = hasBias;
        return this;
    }
}
