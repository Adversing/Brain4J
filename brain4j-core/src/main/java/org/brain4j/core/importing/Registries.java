package org.brain4j.core.importing;

import org.brain4j.core.activation.impl.*;
import org.brain4j.core.clipper.GradientClipper;
import org.brain4j.core.clipper.impl.HardClipper;
import org.brain4j.core.clipper.impl.L2Clipper;
import org.brain4j.core.clipper.impl.NoClipper;
import org.brain4j.core.importing.format.GeneralRegistry;
import org.brain4j.core.layer.Layer;
import org.brain4j.core.layer.impl.*;
import org.brain4j.core.layer.impl.convolutional.ConvLayer;
import org.brain4j.core.layer.impl.convolutional.InputLayer;
import org.brain4j.core.layer.impl.transformer.*;
import org.brain4j.core.layer.impl.utility.ActivationLayer;
import org.brain4j.core.layer.impl.utility.ReshapeLayer;
import org.brain4j.core.layer.impl.utility.SliceLayer;
import org.brain4j.core.layer.impl.utility.SqueezeLayer;
import org.brain4j.math.activation.Activation;

public class Registries {
    
    public static final GeneralRegistry<GradientClipper> CLIPPERS_REGISTRY = new GeneralRegistry<>();
    public static final GeneralRegistry<Activation> ACTIVATION_REGISTRY = new GeneralRegistry<>();
    public static final GeneralRegistry<Layer> LAYER_REGISTRY = new GeneralRegistry<>();
    
    static {
        CLIPPERS_REGISTRY.register("none", NoClipper.class);
        CLIPPERS_REGISTRY.register("hard", HardClipper.class);
        CLIPPERS_REGISTRY.register("l2", L2Clipper.class);
        
        ACTIVATION_REGISTRY.register("elu", ELUActivation.class);
        ACTIVATION_REGISTRY.register("gelu", GELUActivation.class);
        ACTIVATION_REGISTRY.register("leaky_relu", LeakyReLUActivation.class);
        ACTIVATION_REGISTRY.register("linear", LinearActivation.class);
        ACTIVATION_REGISTRY.register("mish", MishActivation.class);
        ACTIVATION_REGISTRY.register("relu", ReLUActivation.class);
        ACTIVATION_REGISTRY.register("sigmoid", SigmoidActivation.class);
        ACTIVATION_REGISTRY.register("softmax", SoftmaxActivation.class);
        ACTIVATION_REGISTRY.register("swish", SwishActivation.class);
        ACTIVATION_REGISTRY.register("tanh", TanhActivation.class);
        
        LAYER_REGISTRY.register("input", InputLayer.class);
        LAYER_REGISTRY.register("dense", DenseLayer.class);
        LAYER_REGISTRY.register("dropout", DropoutLayer.class);
        LAYER_REGISTRY.register("lstm", LSTMLayer.class);
        LAYER_REGISTRY.register("layer_norm", NormLayer.class);
        LAYER_REGISTRY.register("recurrent", RecurrentLayer.class);
        LAYER_REGISTRY.register("conv_2d", ConvLayer.class);
        
        LAYER_REGISTRY.register("embedding", EmbeddingLayer.class);
        LAYER_REGISTRY.register("vocab", OutVocabLayer.class);
        LAYER_REGISTRY.register("positional_encode", PosEncodeLayer.class);
        LAYER_REGISTRY.register("transformer_decoder", TransformerDecoder.class);
        LAYER_REGISTRY.register("transformer_encoder", TransformerEncoder.class);
        
        LAYER_REGISTRY.register("activation", ActivationLayer.class);
        LAYER_REGISTRY.register("reshape", ReshapeLayer.class);
        LAYER_REGISTRY.register("slice", SliceLayer.class);
        LAYER_REGISTRY.register("squeeze", SqueezeLayer.class);
    }
}
