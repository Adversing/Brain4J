package org.brain4j.core.layer.impl.transformer;

import org.brain4j.core.layer.impl.DenseLayer;
import org.brain4j.core.layer.impl.DropoutLayer;
import org.brain4j.core.layer.impl.NormLayer;
import org.brain4j.core.transformer.attention.MaskedMultiHeadAttention;
import org.brain4j.core.transformer.attention.head.AttentionHead;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.index.Range;

import java.util.Arrays;
import java.util.Map;

/**
 * Implements a single decoder block of the Transformer architecture,
 * as introduced in the paper "Attention is All You Need".
 *
 * <p>This class extends the {@link TransformerEncoder} and reuses most of its structure.
 * Unlike the encoder, the decoder applies a causal (triangular) mask in its self-attention layer
 * to prevent attending to future positions during training and inference.
 * This implementation is better fitted for generative models (e.g. GPTs).
 *
 * <p>The expected input shape is a 3D tensor of shape {@code [batch, seq_len, embedding_dim]}.
 * The output has the same shape.
 *
 * @see TransformerEncoder
 * @see MaskedMultiHeadAttention
 * @see DenseLayer
 * @see DropoutLayer
 * @see NormLayer
 * @author xEcho1337
 */
public class TransformerDecoder extends TransformerEncoder {

    private TransformerDecoder() {
        super();
    }
    
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

        Tensor added = attended.add(input);
        Tensor normalized = normalizer1.forward(cache, added);

        Tensor upProjected, downProjected;

        Tensor[] upCache = cache.output(upProjection);
        Tensor[] downCache = cache.output(downProjection);

        int seqLength = input.shape(1);

        if (upCache.length == 0 || downCache.length == 0) {
            upProjected = upProjection.forward(cache, normalized);
            downProjected = downProjection.forward(cache, upProjected);
        } else {
            Tensor cacheUp = upCache[0];
            Tensor cacheDown = downCache[0];
            
            Range[] ranges = { Range.all(), Range.point(seqLength - 1), Range.all() };
            Tensor sliced = normalized.sliceGrad(ranges);

            Tensor upProj = upProjection.forward(cache, sliced);
            Tensor downProj = downProjection.forward(cache, upProj);

            upProjected = cacheUp.concatGrad(upProj, 1);
            downProjected = cacheDown.concatGrad(downProj, 1);
        }

        cache.rememberOutput(upProjection, upProjected);
        cache.rememberOutput(downProjection, downProjected);

        if (cache.training()) {
            downProjected = dropout.forward(cache, downProjected);
        }

        Tensor added2 = downProjected.add(normalized);
        normalized = normalizer2.forward(cache, added2);

        cache.rememberOutput(this, normalized);

        return new Tensor[] { normalized };
    }
}
