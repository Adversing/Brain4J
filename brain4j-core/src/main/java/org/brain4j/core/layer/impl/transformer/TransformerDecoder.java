package org.brain4j.core.layer.impl.transformer;

import org.brain4j.core.layer.impl.DenseLayer;
import org.brain4j.core.layer.impl.DropoutLayer;
import org.brain4j.core.layer.impl.NormLayer;
import org.brain4j.core.transformer.attention.MaskedMultiHeadAttention;
import org.brain4j.core.transformer.attention.MultiHeadAttention;

/**
 * Implements a single decoder block of the Transformer architecture,
 * as introduced in the paper "Attention is All You Need".
 *
 * <p>This class extends the {@link TransformerEncoder} and reuses most of its structure.
 * Unlike the encoder, the decoder applies a causal (triangular) mask in its self-attention layer
 * to prevent attending to future positions during training and inference.
 * This implementation is better fitted for generative models (e.g. GPTs).
 *
 * <p>The expected input shape is a 3D tensor of shape {@code [batch_size, seq_len, embedding_dim]}.
 * The output has the same shape.
 *
 * @see TransformerEncoder
 * @see MaskedMultiHeadAttention
 * @see DenseLayer
 * @see DropoutLayer
 * @see NormLayer
 * @author xEcho1337
 * @since 3.0
 */
public class TransformerDecoder extends TransformerEncoder {
    
    /**
     * Constructs a new decoder block with the specified parameters.
     * @param numHeads the amount of heads in the attention block
     * @param embeddingDim the embedding dimension of the input
     * @param dropout the dropout used when training
     */
    public TransformerDecoder(int numHeads, int embeddingDim, double dropout) {
        super(numHeads, embeddingDim, dropout);
    }

    public MultiHeadAttention createAttention(int heads, int embeddingDim) {
        return new MaskedMultiHeadAttention(clipper, heads, embeddingDim);
    }
}
