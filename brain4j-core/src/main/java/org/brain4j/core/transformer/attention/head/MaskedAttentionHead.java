package org.brain4j.core.transformer.attention.head;

import org.brain4j.common.Tensors;
import org.brain4j.common.tensor.Tensor;
import org.brain4j.core.activation.impl.SoftmaxActivation;
import org.brain4j.core.clipper.GradientClipper;

public class MaskedAttentionHead extends AttentionHead {

    public MaskedAttentionHead(GradientClipper clipper, int embedDimension, int headDimension) {
        super(clipper, embedDimension, headDimension);
    }

    @Override
    public Tensor attend(Tensor input) {
        // input = [batch_size, seq_length, embedding_dim]
        Tensor Q = input.matmulGrad(queryWeights); // [batch_size, seq_length, head_dimension]
        Tensor K = input.matmulGrad(keyWeights); // [batch_size, seq_length, head_dimension]
        Tensor V = input.matmulGrad(valueWeights); // [batch_size, seq_length, head_dimension]
        
        double normalizer = Math.sqrt(headDimension);
        
        // [batch_size, head_dimension, seq_length]
        Tensor K_T = K.transposeGrad();
        // [batch_size, seq_length, seq_length]
        Tensor scores = Q.matmulGrad(K_T).div(normalizer);
        
        int[] shape = scores.shape();
        Tensor mask = Tensors.triangularMask(shape[shape.length - 1]);
        
        Tensor attentionWeights = scores.addGrad(mask).activateGrad(new SoftmaxActivation());
        
        // [batch_size, seq_length, head_dimension]
        return attentionWeights.matmulGrad(V);
    }
}
