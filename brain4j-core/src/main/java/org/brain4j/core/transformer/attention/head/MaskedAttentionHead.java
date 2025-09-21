package org.brain4j.core.transformer.attention.head;

import org.brain4j.math.data.StatesCache;
import org.brain4j.math.Tensors;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.activation.impl.SoftmaxActivation;
import org.brain4j.math.clipper.GradientClipper;
import org.brain4j.math.tensor.index.Range;

public class MaskedAttentionHead extends AttentionHead {

    public MaskedAttentionHead(GradientClipper clipper, int embedDimension, int headDimension) {
        super(clipper, embedDimension, headDimension);
    }

    @Override
    public Tensor attend(Tensor input) {
        // input = [batch, seq_len, embedding_dim]
        Tensor Q = input.matmulGrad(queryWeights); // [batch, seq_len, head_dim]
        Tensor K = input.matmulGrad(keyWeights); // [batch, seq_len, head_dim]
        Tensor V = input.matmulGrad(valueWeights); // [batch, seq_len, head_dim]
        
        double normalizer = Math.sqrt(headDimension);
        
        // [batch, head_dim, seq_len]
        Tensor K_T = K.transposeGrad();
        // [batch, seq_len, seq_len]
        Tensor scores = Q.matmulGrad(K_T).div(normalizer);
        
        int[] shape = scores.shape();
        Tensor mask = Tensors.triangularMask(shape[shape.length - 1]);
        
        Tensor attentionWeights = scores.addGrad(mask).activateGrad(new SoftmaxActivation());
        
        // [batch, seq_len, head_dim]
        return attentionWeights.matmulGrad(V);
    }

    @Override
    public Tensor attend(StatesCache cache, Tensor input) {
        // input = [batch, seq_len, embedding_dim]
        int seqLength = input.shape(1);

        Range[] ranges = { Range.all(), Range.point(seqLength - 1), Range.all() };
        Tensor sliced = input.sliceGrad(ranges); // [batch, 1, embedding_dim]

        Tensor prevK = cache.get(keyWeights);
        Tensor prevV = cache.get(valueWeights);

        Tensor K, V, Q;
        int triangularMaskLength = 1;

        if (prevK == null || prevV == null) {
            K = input.matmulGrad(keyWeights);
            V = input.matmulGrad(valueWeights);
            Q = input.matmulGrad(queryWeights);
            triangularMaskLength = seqLength;
        } else {
            Tensor newK = sliced.matmulGrad(keyWeights);
            Tensor newV = sliced.matmulGrad(valueWeights);
            Q = sliced.matmulGrad(queryWeights); // [batch, 1, head_dim];
            K = prevK.concatGrad(newK, 1);
            V = prevV.concatGrad(newV, 1);
        }

        cache.updateCache(keyWeights, K);
        cache.updateCache(valueWeights, V);

        double normalizer = Math.sqrt(headDimension);

        // [batch, head_dim, seq_len]
        Tensor K_T = K.transposeGrad();
        // [batch, 1 | seq_len, seq_len]
        Tensor scores = Q.matmulGrad(K_T).div(normalizer);

        Tensor mask = Tensors.triangularMask(triangularMaskLength, seqLength);

        // [batch, 1 | seq_len, seq_len]
        Tensor attentionWeights = scores.addGrad(mask).activateGrad(new SoftmaxActivation());
        // [batch, 1 | seq_len, head_dim]
        return attentionWeights.matmulGrad(V);
    }
}
