package org.brain4j.core.layer.impl.transformer;

import com.google.gson.JsonObject;
import org.brain4j.core.layer.Layer;
import org.brain4j.core.layer.impl.DenseLayer;
import org.brain4j.core.layer.impl.DropoutLayer;
import org.brain4j.core.layer.impl.NormLayer;
import org.brain4j.core.training.optimizer.Optimizer;
import org.brain4j.core.training.updater.Updater;
import org.brain4j.core.transformer.attention.head.AttentionHead;
import org.brain4j.math.activation.Activation;
import org.brain4j.math.activation.Activations;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.gpu.device.Device;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.weightsinit.UniformXavierInit;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.random.RandomGenerator;

/**
 * Implements a single encoder block of the Transformer architecture,
 * as introduced in the paper "Attention is All You Need".
 *
 * <p>It includes a multi-head self-attention layer, along with
 * residual connections, layer normalization, feed-forward projection
 * and dropout for regularization.
 *
 * <p>The expected input shape is a 3D tensor of shape {@code [batch, seq_len, embedding_dim]}.
 * The output has the same shape.
 *
 * @see TransformerDecoder
 * @see MultiHeadAttention
 * @see DenseLayer
 * @see DropoutLayer
 * @see NormLayer
 * @author xEcho1337
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
    protected double dropoutRate;

    protected TransformerEncoder() {
    }

    /**
     * Constructs a new encoder block with the specified parameters.
     * @param numHeads the amount of heads in the attention block
     * @param embeddingDim the embedding dimension of the input
     * @param dropout the dropout used when training
     */
    public TransformerEncoder(int numHeads, int embeddingDim, double dropout) {
        this(numHeads, embeddingDim, dropout, Activations.GELU);
    }

    /**
     * Constructs a new encoder block with the specified parameters.
     * @param numHeads the amount of heads in the attention block
     * @param embeddingDim the embedding dimension of the input
     * @param dropout the dropout used when training
     * @param activation the activation used in the projection
     */
    public TransformerEncoder(int numHeads, int embeddingDim, double dropout, Activations activation) {
        this(numHeads, embeddingDim, 4 * embeddingDim, dropout, activation.function());
    }

    /**
     * Constructs a new encoder block with the specified parameters.
     * @param numHeads the amount of heads in the attention block
     * @param embeddingDim the embedding dimension of the input
     * @param projDim the dimension of the projected embedding
     * @param dropout the dropout used when training
     * @param activation the activation used in the projection
     */
    public TransformerEncoder(int numHeads, int embeddingDim, int projDim, double dropout, Activation activation) {
        this.numHeads = numHeads;
        this.embeddingDim = embeddingDim;
        this.dropoutRate = dropout;
        this.activation = activation;
        this.dropout = new DropoutLayer(dropout);
        this.weightInit = new UniformXavierInit();

        this.normalizer1 = new NormLayer();
        this.normalizer2 = new NormLayer();
        this.upProjection = new DenseLayer(projDim);
        this.downProjection = new DenseLayer(embeddingDim);

        this.attention = createAttention(numHeads, embeddingDim);
    }

    public MultiHeadAttention createAttention(int heads, int embeddingDim) {
        return new MultiHeadAttention(clipper, heads, embeddingDim);
    }
    
    @Override
    public void resetGrad() {
        normalizer1.resetGrad();
        normalizer2.resetGrad();
        upProjection.resetGrad();
        downProjection.resetGrad();
        attention.resetGrad();
    }
    
    @Override
    public Layer connect(Layer previous) {
        normalizer1.connect(this);
        normalizer2.connect(this);
        upProjection.connect(this);
        downProjection.connect(upProjection);
        attention.connect(previous);
        return this;
    }

    @Override
    public void initWeights(RandomGenerator generator, int input, int output) {
        normalizer1.initWeights(generator, embeddingDim, embeddingDim);
        normalizer2.initWeights(generator, embeddingDim, embeddingDim);
        upProjection.initWeights(generator, embeddingDim, embeddingDim * 4);
        downProjection.initWeights(generator, embeddingDim * 4, embeddingDim);
        attention.initWeights(generator, embeddingDim, embeddingDim);
    }

    @Override
    public void toDevice(Device device) {
        normalizer1.toDevice(device);
        normalizer2.toDevice(device);
        upProjection.toDevice(device);
        downProjection.toDevice(device);
        attention.toDevice(device);
    }
    
    @Override
    public Tensor[] forward(StatesCache cache, Tensor... inputs) {
        checkInputLength(1, inputs);
        Tensor input = inputs[0];

        if (input.rank() != 3) {
            throw new IllegalArgumentException(
                "Expected input with shape [batch, seq_len, dimension], got: " + Arrays.toString(input.shape())
            );
        }

        Tensor attended = attention.forward(cache, input);
        
        if (cache.training()) {
            attended = dropout.forward(cache, attended);
        }

        Tensor added = attended.addGrad(input);
        Tensor normalized = normalizer1.forward(cache, added);

        Tensor upProjected = upProjection.forward(cache, normalized).activateGrad(activation);
        Tensor downProjected = downProjection.forward(cache, upProjected);

        if (cache.training()) {
            downProjected = dropout.forward(cache, downProjected);
        }

        Tensor added2 = downProjected.addGrad(normalized);
        normalized = normalizer2.forward(cache, added2);

        cache.rememberOutput(this, normalized);
        
        return new Tensor[] { normalized };
    }
    
    @Override
    public void backward(StatesCache cache, Updater updater, Optimizer optimizer) {
        normalizer2.backward(cache, updater, optimizer);
        downProjection.backward(cache, updater, optimizer);
        upProjection.backward(cache, updater, optimizer);
        normalizer1.backward(cache, updater, optimizer);
        attention.backward(cache, updater, optimizer);
    }
    
    @Override
    public int size() {
        return embeddingDim;
    }
    
    @Override
    public void loadWeights(Map<String, Tensor> mappedWeights) {
        this.upProjection = new DenseLayer(0);
        this.downProjection = new DenseLayer(0);
        this.dropout = new DropoutLayer(dropoutRate);
        this.normalizer1 = new NormLayer();
        this.normalizer2 = new NormLayer();
        this.attention = createAttention(numHeads, embeddingDim);

        upProjection.setWeights(mappedWeights.get("up_projection.weights"));
        upProjection.setBias(mappedWeights.get("up_projection.bias"));
        downProjection.setWeights(mappedWeights.get("down_projection.weights"));
        downProjection.setBias(mappedWeights.get("down_projection.bias"));

        normalizer1.setWeights(mappedWeights.get("normalizer_1.weights"));
        normalizer1.setBias(mappedWeights.get("normalizer_1.bias"));
        normalizer2.setWeights(mappedWeights.get("normalizer_2.weights"));
        normalizer2.setBias(mappedWeights.get("normalizer_2.bias"));

        attention.setOutProjWeights(mappedWeights.get("attention.out_proj"));
        attention.setWeights(mappedWeights.get("attention.weights"));
        attention.setBias(mappedWeights.get("attention.bias"));
    }
    
    @Override
    public void serialize(JsonObject object) {
        object.addProperty("dropout", dropoutRate);
        object.addProperty("heads", numHeads);
        object.addProperty("embedding_dim", embeddingDim);
    }
    
    @Override
    public void deserialize(JsonObject object) {
        this.dropoutRate = object.get("dropout").getAsDouble();
        this.numHeads = object.get("heads").getAsInt();
        this.embeddingDim = object.get("embedding_dim").getAsInt();
    }
    
    @Override
    public int totalBiases() {
        return upProjection.totalBiases()
            + downProjection.totalBiases()
            + normalizer1.totalBiases()
            + normalizer2.totalBiases();
    }
    
    @Override
    public int totalWeights() {
        return upProjection.totalWeights()
            + downProjection.totalWeights()
            + normalizer1.totalWeights()
            + normalizer2.totalWeights()
            + attention.totalWeights();
    }
    
    @Override
    public Map<String, Tensor> weightsMap() {
        var result = super.weightsMap();
        
        result.put("up_projection.weights", upProjection.weights());
        result.put("up_projection.bias", upProjection.bias());
        result.put("down_projection.weights", downProjection.weights());
        result.put("down_projection.bias", downProjection.bias());

        result.put("normalizer_1.weights", normalizer1.weights());
        result.put("normalizer_1.bias", normalizer1.bias());
        result.put("normalizer_2.weights", normalizer2.weights());
        result.put("normalizer_2.bias", normalizer2.bias());

        result.put("attention.weights", attention.weights());
        result.put("attention.bias", attention.bias());
        result.put("attention.out_proj", attention.outProjWeights());

        return result;
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
