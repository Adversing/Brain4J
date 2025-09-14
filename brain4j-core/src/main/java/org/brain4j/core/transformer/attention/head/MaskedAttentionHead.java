package org.brain4j.core.transformer.attention.head;

import org.brain4j.core.training.StatesCache;
import org.brain4j.math.Tensors;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.core.activation.impl.SoftmaxActivation;
import org.brain4j.core.clipper.GradientClipper;
import org.brain4j.math.tensor.index.Range;

import java.util.Arrays;

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
        int seqLength = input.shape()[1];

        Range[] ranges = { Range.all(), Range.point(seqLength - 1), Range.all() };
        Tensor sliced = input.sliceGrad(ranges); // [batch, 1, embedding_dim]

        Tensor prevK = cache.keys(this);
        Tensor prevV = cache.values(this);

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

        cache.setKeys(this, K);
        cache.setValues(this, V);
        
        double normalizer = Math.sqrt(headDimension);

        // [batch, head_dim, seq_len]
        Tensor K_T = K.transposeGrad();
        // [batch, 1 | seq_len, seq_len]
        Tensor scores = Q.matmulGrad(K_T).div(normalizer);

        Tensor mask = Tensors.triangularMask(triangularMaskLength, seqLength);

        // [batch, 1 | seq_len, seq_len]
        Tensor attentionWeights = scores.addGrad(mask).activateGrad(new SoftmaxActivation());
        Tensor attentionOutput = attentionWeights.matmulGrad(V); // [batch, 1 | seq_len, head_dim]

        Tensor prev = cache.attentionOutput(this);
        Tensor result = prev == null ? attentionOutput : prev.concatGrad(attentionOutput, 1);

        cache.setAttentionOutput(this, result);
        // [batch, seq_len, head_dim]
        return result;
    }
}
