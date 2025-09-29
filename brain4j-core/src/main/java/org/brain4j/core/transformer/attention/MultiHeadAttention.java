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
public class MultiHeadAttention {

    protected GradientClipper clipper;
    protected List<AttentionHead> heads; // TODO: migrate to single tensor
    protected Tensor outProjWeights;
    protected Tensor qkvWeights;
    protected Tensor qkvBias;
    protected int headCount;
    protected int embeddingDim;
    protected int headDimension;
    
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

        if (embeddingDim % headCount != 0) {
            throw new IllegalArgumentException("Embedding dimension must be divisible by head count! (%s %% %s = %s)"
                    .formatted(embeddingDim, headCount, embeddingDim % headCount));
        }

        this.headDimension = embeddingDim / headCount;
        this.heads = new ArrayList<>();
        this.outProjWeights = Tensors.matrix(embeddingDim, embeddingDim).withGrad();
        this.qkvWeights = Tensors.matrix(embeddingDim, 3 * embeddingDim).withGrad();
        this.qkvBias = Tensors.zeros(3 * embeddingDim).withGrad();

        initializeHeads();
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
    
    /**
     * Transfers all weights and internal tensors to the specified device.
     * @param device the target device, {@code null} if the target is CPU
     */
    public void toDevice(Device device) {
        for (AttentionHead head : heads) {
            head.toDevice(device);
        }

        this.outProjWeights = outProjWeights.to(device);
        this.qkvWeights = qkvWeights.to(device);
    }
    
    /**
     * Initializes the weights of all attention heads and projection matrices.
     * @param generator the random number generator
     * @param weightInit the weight initialization function
     */
    public void initWeights(Random generator, WeightInitialization weightInit) {
        for (AttentionHead head : heads) {
            head.initWeights(generator, weightInit);
        }

        this.outProjWeights.map(_ -> weightInit.generate(generator, embeddingDim, embeddingDim));
        this.qkvWeights.map(_ -> weightInit.generate(generator, embeddingDim, embeddingDim));
    }
    
    /**
     * Computes the forward pass of the multi-head attention mechanism for a given input tensor.
     * <p>This implementation follows the original architecture defined
     * by the <a href="https://arxiv.org/abs/1706.03762">original paper</a>.
     * @param cache cache used to store intermediate results
     * @param input the input tensor of shape {@code [batch, seq_len, embedding_dim]}
     * @return the output tensor of shape {@code [batch, seq_len, embedding_dim]}
     */
    public Tensor attend(StatesCache cache, Tensor input) {
        int batch = input.shape(0);
        int seqLength = input.shape(1);

        // [batch, seq_len, 3 * H * head_dim]
        Tensor QKV = input.matmulGrad(qkvWeights).addGrad(qkvBias);
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

        // [batch, seq_len, embedding_dim]
        Tensor output = context.reshapeGrad(batch, seqLength, embeddingDim);

        // [batch, seq_len, embedding_dim]
        return output.matmulGrad(outProjWeights);
    }
    
    @Deprecated(forRemoval = true)
    protected void initializeHeads() {
        for (int i = 0; i < headCount; i++) {
            heads.add(createAttentionHead());
        }
    }
    
    protected void backward(Updater updater, Optimizer optimizer) {
        optimize(updater, optimizer, outProjWeights);
        optimize(updater, optimizer, qkvWeights);
        optimize(updater, optimizer, qkvBias);
    }
    
    private void optimize(Updater updater, Optimizer optimizer, Tensor tensor) {
        Tensor grad = tensor.grad();
        Tensor optimized = optimizer.step(tensor, grad);
        clipper.clip(optimized);
        updater.change(tensor, optimized);
    }
    
    /**
     * Calculates the total number of trainable weights in this layer.
     * @return the total number of weights
     */
    public int totalWeights() {
        return qkvWeights.elements() + outProjWeights.elements();
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
        for (AttentionHead head : heads()) {
            head.resetGrad();
        }

        outProjWeights.zerograd();
        qkvWeights.zerograd();
        qkvBias.zerograd();
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
    
    public Tensor qkvWeights() {
        return qkvWeights;
    }
    
    public void setQkvWeights(Tensor qkvWeights) {
        this.qkvWeights = qkvWeights;
    }
    
    public Tensor qkvBias() {
        return qkvBias;
    }
    
    public void setQkvBias(Tensor qkvBias) {
        this.qkvBias = qkvBias;
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
