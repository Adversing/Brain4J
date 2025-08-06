package org.brain4j.core.layer.impl.transformer;

import org.brain4j.common.gpu.device.Device;
import org.brain4j.common.tensor.Tensor;
import org.brain4j.core.activation.Activations;
import org.brain4j.core.importing.proto.ProtoModel;
import org.brain4j.core.layer.ForwardContext;
import org.brain4j.core.layer.Layer;
import org.brain4j.core.layer.impl.DenseLayer;
import org.brain4j.core.layer.impl.DropoutLayer;
import org.brain4j.core.layer.impl.NormLayer;
import org.brain4j.core.training.StatesCache;
import org.brain4j.core.training.optimizer.Optimizer;
import org.brain4j.core.training.updater.Updater;
import org.brain4j.core.transformer.attention.MultiHeadAttention;
import org.brain4j.core.weightsinit.UniformXavierInit;

import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * Implements a single encoder block of the Transformer architecture,
 * as introduced in the paper "Attention is All You Need".
 *
 * <p>It includes a multi-head self-attention layer, along with
 * residual connections, layer normalization, feed-forward projection
 * and dropout for regularization.
 *
 * <p>The expected input shape is a 3D tensor of shape {@code [batch_size, seq_len, embedding_dim]}.
 * The output has the same shape.
 *
 * @see TransformerDecoder
 * @see MultiHeadAttention
 * @see DenseLayer
 * @see DropoutLayer
 * @see NormLayer
 * @author xEcho1337
 * @since 3.0
 */
public class TransformerEncoder extends Layer {

    protected DenseLayer upProjection;
    protected DenseLayer downProjection;
    protected NormLayer normalizer;
    protected DropoutLayer dropout;
    protected MultiHeadAttention attention;

    protected int numHeads;
    protected int embeddingDim;
    
    /**
     * Constructs a new encoder block with the specified parameters.
     * @param numHeads the amount of heads in the attention block
     * @param embeddingDim the embedding dimension of the input
     * @param dropout the dropout used when training
     */
    public TransformerEncoder(int numHeads, int embeddingDim, double dropout) {
        this.numHeads = numHeads;
        this.embeddingDim = embeddingDim;
        this.dropout = new DropoutLayer(dropout);
        this.weightInit = new UniformXavierInit();

        this.normalizer = new NormLayer(embeddingDim);
        this.upProjection = new DenseLayer(embeddingDim * 4, Activations.RELU);
        this.downProjection = new DenseLayer(embeddingDim, Activations.LINEAR);

        this.attention = createAttention(numHeads, embeddingDim);
    }

    public MultiHeadAttention createAttention(int heads, int embeddingDim) {
        return new MultiHeadAttention(clipper, heads, embeddingDim);
    }

    @Override
    public Layer connect(Layer previous) {
        this.normalizer.connect(this);
        this.upProjection.connect(this);
        this.downProjection.connect(upProjection);

        return this;
    }

    @Override
    public void initWeights(Random generator, int input, int output) {
        this.normalizer.initWeights(generator, embeddingDim, embeddingDim);
        this.upProjection.initWeights(generator, embeddingDim, embeddingDim * 4);
        this.downProjection.initWeights(generator, embeddingDim * 4, embeddingDim);
        this.attention.compile(generator, weightInit);
    }

    @Override
    public void toDevice(Device device) {
        this.normalizer.toDevice(device);
        this.upProjection.toDevice(device);
        this.downProjection.toDevice(device);
        this.attention.to(device);
    }
    
    @Override
    public List<ProtoModel.Tensor.Builder> serialize(ProtoModel.Layer.Builder layerBuilder) {
        layerBuilder.putAttrs("num_heads", value(numHeads));
        layerBuilder.putAttrs("embedding_dim", value(embeddingDim));
        return List.of();
    }
    
    @Override
    public Tensor forward(ForwardContext context) {
        Tensor input = context.input();

        if (input.rank() != 3) {
            throw new IllegalArgumentException(
                "Input must have shape [batch_size, seq_len, dimension], got: " + Arrays.toString(input.shape())
            );
        }

        int index = context.index();
        boolean training = context.training();

        StatesCache cache = context.cache();
        Tensor attended = attention.attend(cache, input);

        if (training) {
            attended = dropout.forward(new ForwardContext(cache, attended, index, true));
        }

        Tensor added = input.addGrad(attended);
        Tensor normalized = normalizer.forward(new ForwardContext(cache, added, index, true));
        
        Tensor upProjected = upProjection.forward(new ForwardContext(cache, normalized, index, training));
        Tensor downProjected = downProjection.forward(new ForwardContext(cache, upProjected, index, training));

        if (training) {
            downProjected = dropout.forward(new ForwardContext(cache, downProjected, index, true));
        }

        added = normalized.addGrad(downProjected);
        normalized = normalizer.forward(new ForwardContext(cache, added, index, training));

        return normalized;
    }

    @Override
    public void backward(Updater updater, Optimizer optimizer, int index) {
        this.attention.backward(updater, optimizer);
        this.upProjection.backward(updater, optimizer, index);
        this.downProjection.backward(updater, optimizer, index);
    }

    @Override
    public int size() {
        return embeddingDim;
    }

    @Override
    public int totalWeights() {
        return this.upProjection.totalWeights()
            + this.downProjection.totalWeights()
            + this.normalizer.totalWeights()
            + this.attention.totalWeights();
    }
}
