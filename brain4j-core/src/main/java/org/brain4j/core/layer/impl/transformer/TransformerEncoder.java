package org.brain4j.core.layer.impl.transformer;

import org.brain4j.common.gpu.device.Device;
import org.brain4j.common.tensor.Tensor;
import org.brain4j.core.activation.Activations;
import org.brain4j.core.importing.proto.ProtoModel;
import org.brain4j.core.importing.proto.SerializeUtils;
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
    protected NormLayer normalizer1;
    protected NormLayer normalizer2;
    protected DropoutLayer dropout;
    protected MultiHeadAttention attention;

    protected int numHeads;
    protected int embeddingDim;
    
    public TransformerEncoder() {
    }
    
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
        
        this.normalizer1 = new NormLayer(embeddingDim);
        this.normalizer2 = new NormLayer(embeddingDim);
        this.upProjection = new DenseLayer(embeddingDim * 4, Activations.RELU);
        this.downProjection = new DenseLayer(embeddingDim, Activations.LINEAR);

        this.attention = createAttention(numHeads, embeddingDim);
    }

    public MultiHeadAttention createAttention(int heads, int embeddingDim) {
        return new MultiHeadAttention(clipper, heads, embeddingDim);
    }

    @Override
    public Layer connect(Layer previous) {
        normalizer1.connect(this);
        normalizer2.connect(this);
        upProjection.connect(this);
        downProjection.connect(upProjection);

        return this;
    }

    @Override
    public void initWeights(Random generator, int input, int output) {
        normalizer1.initWeights(generator, embeddingDim, embeddingDim);
        normalizer2.initWeights(generator, embeddingDim, embeddingDim);
        upProjection.initWeights(generator, embeddingDim, embeddingDim * 4);
        downProjection.initWeights(generator, embeddingDim * 4, embeddingDim);
        attention.compile(generator, weightInit);
    }

    @Override
    public void toDevice(Device device) {
        normalizer1.toDevice(device);
        normalizer2.toDevice(device);
        upProjection.toDevice(device);
        downProjection.toDevice(device);
        attention.to(device);
    }
    
    @Override
    public void deserialize(List<ProtoModel.Tensor> tensors, ProtoModel.Layer layer) {
        this.numHeads = SerializeUtils.attribute(layer, "num_heads", 0);
        this.embeddingDim = SerializeUtils.attribute(layer, "embedding_dim", 0);
        
        if (!layer.hasTransformer()) {
            throw new IllegalArgumentException("Transformer layer is missing a transformer instance!");
        }
        
        ProtoModel.Transformer transformer = layer.getTransformer();
        
        this.attention = new MultiHeadAttention();
        this.upProjection = new DenseLayer();
        this.downProjection = new DenseLayer();
        this.normalizer1 = new NormLayer();
        this.normalizer2 = new NormLayer();
        this.dropout = new DropoutLayer();
        
        attention.deserialize(transformer.getAttention());
        upProjection.deserialize(SerializeUtils.filterByName(tensors, "up_projection"), transformer.getUpProjection());
        downProjection.deserialize(SerializeUtils.filterByName(tensors, "down_projection"), transformer.getDownProjection());
        normalizer1.deserialize(SerializeUtils.filterByName(tensors, "normalizer_1"), transformer.getNormalizer1());
        normalizer2.deserialize(SerializeUtils.filterByName(tensors, "normalizer_2"), transformer.getNormalizer2());
        dropout.deserialize(List.of(), transformer.getDropout());
    }
    
    @Override
    public void serialize(ProtoModel.Layer.Builder builder) {
        ProtoModel.Transformer.Builder transformerBuilder =
            ProtoModel.Transformer.newBuilder()
                .setAttention(attention.serialize());
        
        ProtoModel.Layer.Builder upProjBuilder = buildLayer(upProjection);
        ProtoModel.Layer.Builder downProjBuilder = buildLayer(downProjection);
        ProtoModel.Layer.Builder dropoutBuilder = buildLayer(dropout);
        ProtoModel.Layer.Builder normalizer1Builder = buildLayer(normalizer1);
        ProtoModel.Layer.Builder normalizer2Builder = buildLayer(normalizer2);
        
        transformerBuilder.setUpProjection(upProjBuilder.build());
        transformerBuilder.setDownProjection(downProjBuilder.build());
        transformerBuilder.setDropout(dropoutBuilder.build());
        transformerBuilder.setNormalizer1(normalizer1Builder.build());
        transformerBuilder.setNormalizer2(normalizer2Builder.build());
        
        builder.putAttrs("num_heads", SerializeUtils.value(numHeads))
            .putAttrs("embedding_dim", SerializeUtils.value(embeddingDim))
            .setTransformer(transformerBuilder.build());
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
        Tensor normalized = normalizer1.forward(new ForwardContext(cache, added, index, true));
        
        Tensor upProjected = upProjection.forward(new ForwardContext(cache, normalized, index, training));
        Tensor downProjected = downProjection.forward(new ForwardContext(cache, upProjected, index, training));

        if (training) {
            downProjected = dropout.forward(new ForwardContext(cache, downProjected, index, true));
        }

        added = normalized.addGrad(downProjected);
        normalized = normalizer2.forward(new ForwardContext(cache, added, index, training));

        cache.setPreActivation(this, normalized);
        
        return normalized;
    }

    @Override
    public void backward(Updater updater, Optimizer optimizer, int index) {
        attention.backward(updater, optimizer);
        upProjection.backward(updater, optimizer, index);
        downProjection.backward(updater, optimizer, index);
    }

    @Override
    public int size() {
        return embeddingDim;
    }

    @Override
    public int totalWeights() {
        return upProjection.totalWeights()
            + downProjection.totalWeights()
            + normalizer1.totalWeights()
            + attention.totalWeights();
    }
    
    @Override
    public List<ProtoModel.Tensor.Builder> weightsList() {
        return List.of(
            SerializeUtils.serializeTensor("normalizer_1.weight", normalizer1.weights()),
            SerializeUtils.serializeTensor("normalizer_1.bias", normalizer1.bias()),
            SerializeUtils.serializeTensor("normalizer_2.weight", normalizer2.weights()),
            SerializeUtils.serializeTensor("normalizer_2.bias", normalizer2.bias()),
            SerializeUtils.serializeTensor("up_projection.weight", upProjection.weights()),
            SerializeUtils.serializeTensor("up_projection.bias", upProjection.bias()),
            SerializeUtils.serializeTensor("down_projection.weight", downProjection.weights()),
            SerializeUtils.serializeTensor("down_projection.bias", downProjection.bias())
        );
    }
    
    private ProtoModel.Layer.Builder buildLayer(Layer layer) {
        ProtoModel.Layer.Builder builder =
            ProtoModel.Layer.newBuilder()
                .setType(layer.getClass().getName())
                .setBasic(ProtoModel.BasicLayer.newBuilder()
                    .setDimension(layer.size()));
    
        layer.serialize(builder);
        return builder;
    }
    
    public DenseLayer upProjection() {
        return upProjection;
    }
    
    public DenseLayer downProjection() {
        return downProjection;
    }
    
    public NormLayer normalizer1() {
        return normalizer1;
    }
    
    public NormLayer normalizer2() {
        return normalizer2;
    }
    
    public DropoutLayer dropout() {
        return dropout;
    }
    
    public MultiHeadAttention attention() {
        return attention;
    }
    
    public int numHeads() {
        return numHeads;
    }
    
    public int embeddingDim() {
        return embeddingDim;
    }
}
