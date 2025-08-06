package org.brain4j.core.layer.impl.transformer;

import org.brain4j.common.Tensors;
import org.brain4j.common.tensor.Tensor;
import org.brain4j.core.importing.proto.ProtoModel;
import org.brain4j.core.layer.ForwardContext;
import org.brain4j.core.layer.Layer;
import org.brain4j.core.weightsinit.UniformXavierInit;

import java.util.Arrays;
import java.util.List;
import java.util.Random;

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

    private final int vocabSize;
    private final int embeddingDim;

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
    public Layer connect(Layer previous) {
        this.weights = Tensors.zeros(vocabSize, embeddingDim).withGrad();
        return this;
    }

    @Override
    public void initWeights(Random generator, int input, int output) {
        this.weights.map(x -> weightInit.generate(generator, input, output));
    }
    
    @Override
    public List<ProtoModel.Tensor.Builder> serialize(ProtoModel.Layer.Builder layerBuilder) {
        layerBuilder.putAttrs("vocab_size", value(vocabSize));
        layerBuilder.putAttrs("embedding_dim", value(embeddingDim));
        return List.of(serializeTensor("weight", weights));
    }
    
    @Override
    public Tensor forward(ForwardContext context) {
        Tensor input = context.input();
        int[] shape = input.shape();

        if (shape.length != 2) {
            throw new IllegalStateException(
                    "Expecting shape [batch_size, seq_length] with dimension 2, got " + Arrays.toString(shape)
            );
        }

        int batchSize = shape[0];
        int seqLength = shape[1];
        
        Tensor oneHot = Tensors.zeros(batchSize, seqLength, vocabSize).withGrad();
        
        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLength; s++) {
                int tokenId = (int) input.get(b, s);
                oneHot.set(1.0f, b, s, tokenId);
            }
        }
        
        // [batch_size, seq_length, embedding_dim]
        return oneHot.matmulGrad(weights);
    }

    @Override
    public int size() {
        return embeddingDim;
    }
}
