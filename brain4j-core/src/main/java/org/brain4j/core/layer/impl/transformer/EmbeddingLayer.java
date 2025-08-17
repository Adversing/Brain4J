package org.brain4j.core.layer.impl.transformer;

import org.brain4j.common.Tensors;
import org.brain4j.common.tensor.Tensor;
import org.brain4j.core.importing.proto.ProtoModel;
import org.brain4j.core.importing.proto.SerializeUtils;
import org.brain4j.core.layer.Layer;
import org.brain4j.core.training.StatesCache;
import org.brain4j.core.training.optimizer.Optimizer;
import org.brain4j.core.training.updater.Updater;
import org.brain4j.core.weightsinit.UniformXavierInit;

import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.stream.IntStream;

/**
 * Embedding layer implementation for transformer architectures.
 * <p>
 * This layer maps integer token indices to dense embedding vectors.
 * It expects an input tensor of shape <code>[batch_size, seq_length]</code> where
 * each element is a token ID located in the vocabulary.
 * </p>
 * <p>
 * The output is a tensor of shape <code>[batch_size, seq_length, embedding_dim]</code>,
 * where each token index is replaced by its corresponding embedding vector.
 * </p>
 * @since 3.0
 * @author xEcho1337
 */
public class EmbeddingLayer extends Layer {

    private int vocabSize;
    private int embeddingDim;
    
    public EmbeddingLayer() {
    }
    
    /**
     * Constructs a new instance of an embedding layer.
     * @param vocabSize the vocabulary size
     * @param embeddingDim the embedding dimension
     */
    public EmbeddingLayer(int vocabSize, int embeddingDim) {
        this.vocabSize = vocabSize;
        this.embeddingDim = embeddingDim;
        this.weightInit = new UniformXavierInit();
    }

    @Override
    public Tensor forward(StatesCache cache, Tensor input) {
        int[] shape = input.shape();

        if (shape.length != 2) {
            throw new IllegalStateException(
                "Expecting shape [batch_size, seq_length] with dimension 2, got " + Arrays.toString(shape)
            );
        }

        int batchSize = shape[0];
        int seqLength = shape[1];

        Tensor output = Tensors.zeros(batchSize, seqLength, embeddingDim).withGrad();

        float[] outData = output.data();
        float[] weightData = weights.data();

        IntStream.range(0, batchSize)
            .parallel()
            .forEach(b -> {
                for (int s = 0; s < seqLength; s++) {
                    int tokenId = (int) input.get(b, s);
                    int outOffset = (b * seqLength + s) * embeddingDim;
                    int weightOffset = tokenId * embeddingDim;

                    System.arraycopy(weightData, weightOffset, outData, outOffset, embeddingDim);
                }
            });

        cache.rememberInput(this, input);
        cache.rememberOutput(this, output);

        // [batch_size, seq_length, embedding_dim]
        return output;
    }

    @Override
    public void backward(StatesCache cache, Updater updater, Optimizer optimizer) {
        Tensor input = cache.input(this);
        Tensor output = cache.output(this);
        Tensor gradOutput = output.grad();
        
        int[] shape = output.shape();

        int batchSize = shape[0];
        int seqLength = shape[1];

        Tensor weightsGrad = weights.grad();

        if (weightsGrad == null) {
            weightsGrad = Tensors.zeros(weights.shape());
        }

        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLength; s++) {
                int tokenId = (int) input.get(b, s);

                for (int d = 0; d < embeddingDim; d++) {
                    float gradient = gradOutput.get(b, s, d);
                    weightsGrad.set(gradient, tokenId, d);
                }
            }
        }

        Tensor optimized = optimizer.step(weights, weightsGrad);

        clipper.clip(optimized);
        updater.change(weights, optimized);
    }

    @Override
    public Layer connect(Layer previous) {
        this.weights = Tensors.zeros(vocabSize, embeddingDim).withGrad();
        return this;
    }

    @Override
    public void initWeights(Random generator, int input, int output) {
        this.weights.map(x -> weightInit.generate(generator, input, output));
    }
    
    @Override
    public void deserialize(List<ProtoModel.Tensor> tensors, ProtoModel.Layer layer) {
        this.vocabSize = SerializeUtils.attribute(layer, "vocab_size", 0);
        this.embeddingDim = SerializeUtils.attribute(layer, "embedding_dim", 0);
        
        for (ProtoModel.Tensor tensor : tensors) {
            if (tensor.getName().contains("weight")) {
                this.weights = SerializeUtils.deserializeTensor(tensor);
            }
        }
    }
    
    @Override
    public void serialize(ProtoModel.Layer.Builder builder) {
        builder.putAttrs("vocab_size", SerializeUtils.value(vocabSize));
        builder.putAttrs("embedding_dim", SerializeUtils.value(embeddingDim));
    }
    
    @Override
    public List<ProtoModel.Tensor.Builder> weightsList() {
        return List.of(
            SerializeUtils.serializeTensor("weight", weights)
        );
    }

    @Override
    public int size() {
        return embeddingDim;
    }
}
