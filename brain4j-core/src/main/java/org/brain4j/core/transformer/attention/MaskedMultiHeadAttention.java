package org.brain4j.core.transformer.attention;

import org.brain4j.core.clipper.GradientClipper;
import org.brain4j.core.transformer.attention.head.AttentionHead;
import org.brain4j.core.transformer.attention.head.MaskedAttentionHead;

public class MaskedMultiHeadAttention extends MultiHeadAttention {

    public MaskedMultiHeadAttention(GradientClipper clipper, int headCount, int modelDimension) {
        super(clipper, headCount, modelDimension);
    }

    @Override
    public AttentionHead createAttentionHead() {
        return new MaskedAttentionHead(clipper, embeddingDim, headDimension);
    }
}
