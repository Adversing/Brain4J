package org.brain4j.core.transformer.attention;

import org.brain4j.core.layer.impl.transformer.MultiHeadAttention;
import org.brain4j.math.Tensors;
import org.brain4j.math.activation.impl.SoftmaxActivation;
import org.brain4j.math.clipper.GradientClipper;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.index.Range;

import java.util.Arrays;

public class MaskedMultiHeadAttention extends MultiHeadAttention {
    
    public MaskedMultiHeadAttention(GradientClipper clipper, int headCount, int modelDimension) {
        super(clipper, headCount, modelDimension);
    }
    
    @Override
    public Tensor[] forward(StatesCache cache, Tensor... inputs) {
        Tensor input = inputs[0];
        int batch = input.shape(0);
        int seqLength = input.shape(1);

        Range[] slicingRanges = new Range[] { Range.all(), Range.point(seqLength - 1), Range.all() }; // [batch, 1, dim]
        Tensor cachedOutput = cache.get(outProj);
        Tensor cachedQKV = cache.get(weights);
        Tensor QKV; // [batch, seq_len, 3 * H * head_dim]
        
        if (cachedQKV != null) {
            Tensor newTokens = input.sliceGrad(slicingRanges);
            Tensor proj = newTokens.matmulGrad(weights);
            
            QKV = cachedQKV.concatGrad(proj, 1);
        } else QKV = input.matmulGrad(weights);
        
        cache.set(weights, QKV);

        if (attnQkvHasBias) QKV = QKV.addGrad(bias);

        // [batch, heads, seq_len, 3, head_dim]
        int D = embeddingDim;
        int H = headCount;
        int d = headDimension;

        Range all = Range.all();
        Tensor Q = QKV.sliceGrad(all, all, Range.interval(0, D));
        Tensor K = QKV.sliceGrad(all, all, Range.interval(D, 2 * D));
        Tensor V = QKV.sliceGrad(all, all, Range.interval(2 * D, 3 * D));

        // [batch, heads, seq_len, head_dim]
        Q = Q.reshapeGrad(batch, seqLength, H, d).transposeGrad(1, 2);
        K = K.reshapeGrad(batch, seqLength, H, d).transposeGrad(1, 2);
        V = V.reshapeGrad(batch, seqLength, H, d).transposeGrad(1, 2);

        double normalizer = Math.sqrt(headDimension);

        Tensor mask = Tensors.triangularMask(seqLength, seqLength);

        // [batch, heads, head_dim, seq_len]
        Tensor K_T = K.transposeGrad();
        // [batch, heads, seq_len, seq_len]
        Tensor scores = Q.matmulGrad(K_T).div(normalizer);
        Tensor attentionMap = scores.addGrad(mask);
        Tensor probabilities = attentionMap.activateGrad(new SoftmaxActivation());
        // [batch, heads, seq_len, head_dim]
        Tensor context = probabilities.matmulGrad(V);
        // [batch, seq_len, heads, head_dim]
        context = context.transposeGrad(1, 2);
        
        // [batch, seq_len, embedding_dim]
        Tensor output = context.reshapeGrad(batch, seqLength, embeddingDim);
        Tensor result;
        
        if (cachedOutput != null) {
            Tensor newOutput = output.sliceGrad(slicingRanges);
            Tensor proj = newOutput.matmulGrad(outProj);
            
            result = cachedOutput.concatGrad(proj, 1);
        } else result = output.matmulGrad(outProj);

        cache.set(outProj, result);

        if (attnOutHasBias) result = result.addGrad(outBias);

        return new Tensor[] { result };
    }
}
