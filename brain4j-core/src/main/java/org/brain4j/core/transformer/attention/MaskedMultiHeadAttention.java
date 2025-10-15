package org.brain4j.core.transformer.attention;

import org.brain4j.core.layer.impl.transformer.MultiHeadAttention;
import org.brain4j.math.Tensors;
import org.brain4j.math.activation.impl.SoftmaxActivation;
import org.brain4j.math.clipper.GradientClipper;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.index.Range;

public class MaskedMultiHeadAttention extends MultiHeadAttention {
    
    private final Object keyCache = new Object();
    private final Object valueCache = new Object();
    
    public MaskedMultiHeadAttention(GradientClipper clipper, int headCount, int modelDimension) {
        super(clipper, headCount, modelDimension);
    }
    
    @Override
    public Tensor[] forward(StatesCache cache, Tensor... inputs) {
        Tensor input = inputs[0];
        int batch = input.shape(0);
        int seqLength = input.shape(1);
        
        Tensor prevK = cache.get(keyCache);
        Tensor prevV = cache.get(valueCache);
        
        Tensor Q;
        Tensor Kcat;
        Tensor Vcat;
        int seqQLen;
        
//        if (prevK != null && prevV != null) {
//            Tensor newInput = input.sliceGrad(Range.all(), Range.point(seqLength - 1), Range.all()); // [batch, 1, embeddingDim]
//            Tensor newQKV = newInput.matmulGrad(weights);
//
//            if (attnQkvHasBias) {
//                newQKV = newQKV.addGrad(bias);
//            }
//
//            int newSeqLen = 1;
//            Tensor reshapedNew = newQKV.reshapeGrad(batch, newSeqLen, headCount, 3, headDimension).transposeGrad(1, 2);
//
//            Range all = Range.all();
//            Tensor[] newQKVs = new Tensor[3];
//
//            for (int i = 0; i < 3; i++) {
//                Tensor tmp = reshapedNew.sliceGrad(all, all, all, Range.point(i), all);
//                newQKVs[i] = tmp.squeezeGrad(3); // [batch, heads, newSeqLen, head_dim]
//            }
//
//            Q = newQKVs[0];
//            Tensor Knew = newQKVs[1];
//            Tensor Vnew = newQKVs[2];
//
//            Kcat = prevK.concatGrad(Knew, 2);
//            Vcat = prevV.concatGrad(Vnew, 2);
//
//            seqQLen = Q.shape(2);
//        } else {
            Tensor QKV = input.matmulGrad(weights);
            
            if (attnQkvHasBias) {
                QKV = QKV.addGrad(bias);
            }
            
            Tensor reshaped = QKV.reshapeGrad(batch, seqLength, headCount, 3, headDimension).transposeGrad(1, 2);
            
            Range all = Range.all();
            Tensor[] QKVs = new Tensor[3];
            
            for (int i = 0; i < 3; i++) {
                Tensor tmp = reshaped.sliceGrad(all, all, all, Range.point(i), all);
                QKVs[i] = tmp.squeezeGrad(3);
            }
            
            Q = QKVs[0];
            Kcat = QKVs[1];
            Vcat = QKVs[2];
            seqQLen = Q.shape(2);
//        }
        
        cache.updateCache(keyCache, Kcat);
        cache.updateCache(valueCache, Vcat);
        
        double normalizer = Math.sqrt(headDimension);
        
        Tensor K_T = Kcat.transposeGrad();
        Tensor scores = Q.matmulGrad(K_T).div(normalizer);
        
        int totalSeq = Kcat.shape(2);
        Tensor mask = Tensors.triangularMask(seqQLen, totalSeq);
        
        Tensor attentionWeights = scores.addGrad(mask).activateGrad(new SoftmaxActivation());
        Tensor context = attentionWeights.matmulGrad(Vcat); // [batch, heads, seq_q, head_dim]
        
        context = context.transposeGrad(1, 2); // [batch, seq_q, heads, head_dim]
        Tensor output = context.reshapeGrad(batch, seqQLen, embeddingDim); // [batch, seq_q, embedDim]
        Tensor projected = output.matmulGrad(outProj);
        
        return new Tensor[] { projected };
    }
}
