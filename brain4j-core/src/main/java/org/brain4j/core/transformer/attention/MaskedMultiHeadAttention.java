package org.brain4j.core.transformer.attention;

import org.brain4j.core.layer.impl.transformer.MultiHeadAttention;
import org.brain4j.core.transformer.attention.head.AttentionHead;
import org.brain4j.core.transformer.attention.head.MaskedAttentionHead;
import org.brain4j.math.Tensors;
import org.brain4j.math.clipper.GradientClipper;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.tensor.Tensor;

import java.util.List;

public class MaskedMultiHeadAttention extends MultiHeadAttention {

    public MaskedMultiHeadAttention(GradientClipper clipper, int headCount, int modelDimension) {
        super(clipper, headCount, modelDimension);
    }

    @Override
    public AttentionHead createAttentionHead() {
        return new MaskedAttentionHead(clipper, embeddingDim, headDimension);
    }

    @Override
    public Tensor[] forward(StatesCache cache, Tensor[] inputs) {
        Tensor input = inputs[0];
        Tensor[] outputs = new Tensor[heads.size()];

        for (int i = 0; i < heads.size(); i++) {
            // [batch, 1 | seq_len, head_dim]
            outputs[i] = heads.get(i).attend(cache, input);
        }

        // [batch, 1 | seq_len, embedding_dim]
        Tensor result = Tensors.concatGrad(List.of(outputs));
        Tensor projected = result.matmulGrad(outProj);
        Tensor prev = cache.get(this);

        if (prev != null) {
            projected = prev.concatGrad(projected, 1);
        }

        cache.updateCache(this, projected);
        return new Tensor[] { projected };
    }
}
