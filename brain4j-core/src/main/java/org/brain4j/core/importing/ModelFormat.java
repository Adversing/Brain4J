package org.brain4j.core.importing;

import org.brain4j.common.tensor.autograd.Operation;
import org.brain4j.common.tensor.autograd.impl.*;
import org.brain4j.core.activation.impl.*;
import org.brain4j.core.layer.Layer;
import org.brain4j.core.layer.impl.*;
import org.brain4j.core.layer.impl.convolutional.ConvLayer;
import org.brain4j.core.layer.impl.convolutional.InputLayer;
import org.brain4j.core.layer.impl.transformer.*;
import org.brain4j.core.layer.impl.utility.ActivationLayer;
import org.brain4j.core.layer.impl.utility.ReshapeLayer;
import org.brain4j.core.layer.impl.utility.SliceLayer;
import org.brain4j.core.layer.impl.utility.SqueezeLayer;
import org.brain4j.core.model.Model;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Supplier;

public interface ModelFormat {

    Map<String, Operation> OPERATION_MAP = new HashMap<>() {
        {
            put("Add", new AddOperation());
            put("Sub", new SubOperation());
            put("Mul", new MulOperation());
            put("Div", new DivOperation());
            put("Gemm", new GemmOperation());
            put("MatMul", new MatMulOperation());

            put("Relu", new ActivationOperation(new ReLUActivation()));
            put("Sigmoid", new ActivationOperation(new SigmoidActivation()));
            put("Tanh", new ActivationOperation(new TanhActivation()));
            put("LeakyRelu", new ActivationOperation(new LeakyReLUActivation()));
            put("Gelu", new ActivationOperation(new GELUActivation()));
            put("Softmax", new ActivationOperation(new SoftmaxActivation()));

            put("LayerNormalization", new LayerNormOperation( 1e-5));
        }
    };

    Map<String, Class<? extends Layer>> LAYER_IDENTITY_MAP = new HashMap<>() {
        {
            put("dense", DenseLayer.class);
            put("dropout", DropoutLayer.class);
            put("lstm", LSTMLayer.class);
            put("layer_norm", NormLayer.class);
            put("recurrent", RecurrentLayer.class);

            put("conv_2d", ConvLayer.class);
            put("conv_input", InputLayer.class);

            put("embedding", EmbeddingLayer.class);
            put("vocab", OutVocabLayer.class);
            put("positional_encode", PosEncodeLayer.class);
            put("transformer_decoder", TransformerDecoder.class);
            put("transformer_encoder", TransformerEncoder.class);

            put("activation", ActivationLayer.class);
            put("reshape", ReshapeLayer.class);
            put("slice", SliceLayer.class);
            put("squeeze", SqueezeLayer.class);
        }
    };
    
    <T extends Model> T deserialize(byte[] bytes, Supplier<T> constructor) throws Exception;
    
    void serialize(Model model, File file) throws IOException;
}
