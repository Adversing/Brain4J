package org.brain4j.core.importing;

import org.brain4j.math.activation.impl.*;
import org.brain4j.math.clipper.GradientClipper;
import org.brain4j.math.clipper.impl.HardClipper;
import org.brain4j.math.clipper.impl.L2Clipper;
import org.brain4j.math.clipper.impl.NoClipper;
import org.brain4j.core.importing.format.GeneralRegistry;
import org.brain4j.core.importing.onnx.ProtoOnnx.*;
import org.brain4j.core.layer.Layer;
import org.brain4j.core.layer.impl.*;
import org.brain4j.core.layer.impl.convolutional.ConvLayer;
import org.brain4j.core.layer.impl.utility.*;
import org.brain4j.core.layer.impl.transformer.*;
import org.brain4j.core.loss.LossFunction;
import org.brain4j.core.loss.impl.*;
import org.brain4j.core.training.optimizer.Optimizer;
import org.brain4j.core.training.optimizer.impl.Adam;
import org.brain4j.core.training.optimizer.impl.AdamW;
import org.brain4j.core.training.optimizer.impl.GradientDescent;
import org.brain4j.core.training.optimizer.impl.Lion;
import org.brain4j.core.training.updater.Updater;
import org.brain4j.core.training.updater.impl.NormalUpdater;
import org.brain4j.core.training.updater.impl.StochasticUpdater;
import org.brain4j.math.activation.Activation;
import org.brain4j.math.tensor.autograd.Operation;
import org.brain4j.math.tensor.autograd.impl.*;

public class Registries {
    
    public static final GeneralRegistry<Operation, NodeProto> ONNX_OPERATIONS_REGISTRY = new GeneralRegistry<>();
    public static final GeneralRegistry<Optimizer, Object> OPTIMIZERS_REGISTRY = new GeneralRegistry<>();
    public static final GeneralRegistry<LossFunction, Object> LOSS_FUNCTION_REGISTRY = new GeneralRegistry<>();
    public static final GeneralRegistry<Updater, Object> UPDATERS_REGISTRY = new GeneralRegistry<>();
    public static final GeneralRegistry<GradientClipper, Object> CLIPPERS_REGISTRY = new GeneralRegistry<>();
    public static final GeneralRegistry<Activation, Object> ACTIVATION_REGISTRY = new GeneralRegistry<>();
    public static final GeneralRegistry<Layer, Object> LAYER_REGISTRY = new GeneralRegistry<>();
    
    static {
        ONNX_OPERATIONS_REGISTRY.register("Add", AddOperation.class);
        ONNX_OPERATIONS_REGISTRY.register("Add", AddOperation.class);
        ONNX_OPERATIONS_REGISTRY.register("Sub", SubOperation.class);
        ONNX_OPERATIONS_REGISTRY.register("Mul", MulOperation.class);
        ONNX_OPERATIONS_REGISTRY.register("Div", DivOperation.class);
        ONNX_OPERATIONS_REGISTRY.register("Gemm", GemmOperation.class);
        ONNX_OPERATIONS_REGISTRY.register("MatMul", MatMulOperation.class);

        ONNX_OPERATIONS_REGISTRY.register("Concat", (node) -> {
            int axis = (int) node.getAttribute(0).getI();
            return new ConcatOperation(axis);
        });
        ONNX_OPERATIONS_REGISTRY.register("Squeeze", (node) -> {
            int dimension = (int) node.getAttribute(0).getI();
            return new SqueezeOperation(dimension);
        });
        ONNX_OPERATIONS_REGISTRY.register("Concat", ConcatOperation.class);
        ONNX_OPERATIONS_REGISTRY.register("Relu", (_) -> new ActivationOperation(new ReLUActivation()));
        ONNX_OPERATIONS_REGISTRY.register("Relu", ActivationOperation.class);
        ONNX_OPERATIONS_REGISTRY.register("Sigmoid", (_) -> new ActivationOperation(new SigmoidActivation()));
        ONNX_OPERATIONS_REGISTRY.register("Sigmoid", ActivationOperation.class);
        ONNX_OPERATIONS_REGISTRY.register("Tanh", (_) -> new ActivationOperation(new TanhActivation()));
        ONNX_OPERATIONS_REGISTRY.register("Tanh", ActivationOperation.class);
        ONNX_OPERATIONS_REGISTRY.register("LeakyRelu", (_) -> new ActivationOperation(new LeakyReLUActivation()));
        ONNX_OPERATIONS_REGISTRY.register("LeakyRelu", ActivationOperation.class);
        ONNX_OPERATIONS_REGISTRY.register("Gelu", (_) -> new ActivationOperation(new GELUActivation()));
        ONNX_OPERATIONS_REGISTRY.register("Gelu", ActivationOperation.class);
        ONNX_OPERATIONS_REGISTRY.register("Softmax", (_) -> new ActivationOperation(new SoftmaxActivation()));
        ONNX_OPERATIONS_REGISTRY.register("Softmax", ActivationOperation.class);
        ONNX_OPERATIONS_REGISTRY.register("LayerNormalization", (node) -> {
            float epsilon = node.getAttribute(0).getF();
            return new LayerNormOperation( epsilon);
        });
        
        OPTIMIZERS_REGISTRY.register("adam", Adam.class);
        OPTIMIZERS_REGISTRY.register("adamw", AdamW.class);
        OPTIMIZERS_REGISTRY.register("gradient_descent", GradientDescent.class);
        OPTIMIZERS_REGISTRY.register("lion", Lion.class);
        
        UPDATERS_REGISTRY.register("stochastic", StochasticUpdater.class);
        UPDATERS_REGISTRY.register("normal", NormalUpdater.class);
        
        LOSS_FUNCTION_REGISTRY.register("binary_cross_entropy", BinaryCrossEntropy.class);
        LOSS_FUNCTION_REGISTRY.register("cosine_embedding", CosineEmbedding.class);
        LOSS_FUNCTION_REGISTRY.register("cross_entropy", CrossEntropy.class);
        LOSS_FUNCTION_REGISTRY.register("huber_loss", HuberLoss.class);
        LOSS_FUNCTION_REGISTRY.register("mean_absolute_error", MeanAbsoluteError.class);
        LOSS_FUNCTION_REGISTRY.register("mean_squared_error", MeanSquaredError.class);
        
        CLIPPERS_REGISTRY.register("none", NoClipper.class);
        CLIPPERS_REGISTRY.register("clamp", HardClipper.class);
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
        LAYER_REGISTRY.register("positional_encode", PosEncodeLayer.class);
        LAYER_REGISTRY.register("transformer_decoder", TransformerDecoder.class);
        LAYER_REGISTRY.register("transformer_encoder", TransformerEncoder.class);
        
        LAYER_REGISTRY.register("activation", ActivationLayer.class);
        LAYER_REGISTRY.register("reshape", ReshapeLayer.class);
        LAYER_REGISTRY.register("slice", SliceLayer.class);
        LAYER_REGISTRY.register("squeeze", SqueezeLayer.class);
    }
}
