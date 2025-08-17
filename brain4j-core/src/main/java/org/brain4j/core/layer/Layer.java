package org.brain4j.core.layer;

import org.brain4j.math.Tensors;
import org.brain4j.math.activation.Activation;
import org.brain4j.math.gpu.device.Device;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.weightsinit.WeightInitialization;
import org.brain4j.core.activation.impl.LinearActivation;
import org.brain4j.core.clipper.GradientClipper;
import org.brain4j.core.clipper.impl.HardClipper;
import org.brain4j.core.importing.proto.ProtoModel;
import org.brain4j.core.importing.proto.SerializeUtils;
import org.brain4j.core.loss.LossFunction;
import org.brain4j.core.training.StatesCache;
import org.brain4j.core.training.optimizer.Optimizer;
import org.brain4j.core.training.updater.Updater;

import java.util.List;
import java.util.Random;

/**
 * Abstract base class for all neural network layers.
 * Each layer processes input tensors, computes forward and backward passes independently,
 * and holds its own parameters such as weights, biases, activation function and gradient clipper.
 * @author xEcho1337
 */
public abstract class Layer {

    protected Activation activation = new LinearActivation();
    protected GradientClipper clipper = new HardClipper(5);
    protected WeightInitialization weightInit = activation.defaultWeightInit();

    protected Tensor weights;
    protected Tensor bias;
    
    public Layer() {
    }
    
    /**
     * Deserializes the weights and the layer attributes.
     * @param tensors the layer weights
     * @param layer the layer instance, containing attributes
     */
    public void deserialize(List<ProtoModel.Tensor> tensors, ProtoModel.Layer layer) {
    }
    
    /**
     * Serializes this layer in the provided builder.
     * @param builder the builder that will store the data
     */
    public void serialize(ProtoModel.Layer.Builder builder) {
        ProtoModel.Clipper.Builder clipperBuilder = ProtoModel.Clipper.newBuilder();
        
        clipper.serialize(clipperBuilder);
        
        builder.setClipper(clipperBuilder);
    }
    
    /**
     * Performs a forward pass through this layer.
     *
     * @param cache the states cache for this forward pass
     * @param input the input tensor
     * @return the output tensor
     */
    public abstract Tensor forward(StatesCache cache, Tensor input);

    /**
     * Returns the output size of this layer, i.e. the number of neurons.
     * @return the output size
     */
    public abstract int size();

    /**
     * Constructs the tensors for weights in this layer.
     * @param previous the previous layer in the model
     * @return this layer by default
     */
    public Layer connect(Layer previous) {
        return this;
    }

    /**
     * Initializes the previously constructed weights with random values.
     * @param generator the random number generator
     * @param input the input dimension
     * @param output the output dimension
     */
    public void initWeights(Random generator, int input, int output) {
        // No-op
    }

    /**
     * Ports the weights of this layer to the specified device memory.
     * @param device the device to port the weights on
     */
    public void toDevice(Device device) {
        if (weights != null) {
            weights = weights.to(device).withGrad();
        }

        if (bias != null) {
            bias = bias.to(device).withGrad();
        }
    }

    /**
     * Computes the loss (the gradient) with respect to the loss function and launches the autograd.
     * This method should only be called for the last layer of the neural network.
     *
     * @param cache the state cache of this inference
     * @param targets the target tensor
     * @param outputs the output tensor
     * @param lossFunction the loss function of this model
     */
    public void computeLoss(
        StatesCache cache,
        Tensor[] targets,
        Tensor[] outputs,
        LossFunction lossFunction
    ) {
        Tensor preOutput = cache.output(this);

        if (targets.length != outputs.length) {
            throw new IllegalArgumentException("Targets amount does not equal to output amount.");
        }

        Tensor totalDelta = null;

        for (int i = 0; i < outputs.length; i++) {
            Tensor output = outputs[i];
            Tensor target = targets[i];

            Tensor error = output.minus(target);
            Tensor derivatives = activation.derivative(preOutput);

            Tensor delta = lossFunction.delta(error, derivatives);

            if (totalDelta == null) {
                totalDelta = Tensors.zeros(delta.shape());
            }

            totalDelta.add(delta);
        }

        preOutput.backward(totalDelta);
    }

    /**
     * Computes the backward step for this layer, by calling the optimizer and scheduling weights update.
     *
     * @param cache the states cache of the forward pass
     * @param updater the updater of this model
     * @param optimizer the optimizer of this model
     */
    public void backward(StatesCache cache, Updater updater, Optimizer optimizer) {
        if (weights != null && weights.grad() != null) {
            Tensor weightsGrad = optimizer.step(weights, weights.grad());
            
            clipper.clip(weightsGrad);
            updater.change(weights, weightsGrad);
        }
        
        if (bias != null && bias.grad() != null) {
            Tensor biasGrad = bias.grad().sum(0, false);
            
            clipper.clip(biasGrad);
            updater.change(bias, biasGrad);
        }
    }

    /**
     * Validates if the input can be passed as an input to this layer.
     * This is done by checking the input dimension and comparing it
     * to the layer's expected dimension.
     *
     * @param input the input tensor
     * @return <code>true</code> if the input is valid, <code>false</code> otherwise
     */
    public boolean validateInput(Tensor input) {
        return true;
    }

    /**
     * Gets the activation function for this layer.
     * @return the activation function
     */
    public Activation activation() {
        return activation;
    }

    /**
     * Gets the gradient clipping function for this layer.
     * @return the gradient clipping function
     */
    public GradientClipper clipper() {
        return clipper;
    }

    /**
     * Sets the gradient clipping function for this layer.
     * @param clipper the new gradient clipping function
     * @return this layer
     */
    public Layer clipper(GradientClipper clipper) {
        this.clipper = clipper;
        return this;
    }

    /**
     * Gets the weight initialization function for this layer.
     * @return the weight initialization function
     */
    public WeightInitialization weightInit() {
        return weightInit;
    }

    /**
     * Sets the weight initialization function for this layer.
     * @param weightInit the new weight initialization function
     * @return this layer
     */
    public Layer weightInit(WeightInitialization weightInit) {
        this.weightInit = weightInit;
        return this;
    }

    /**
     * Resets the gradients for all the weights in this layer.
     */
    public void resetGrad() {
        if (weights != null) {
            weights.zerograd();
        }

        if (bias != null) {
            bias.zerograd();
        }
    }

    /**
     * Gets the weights of this layer.
     * @return the weights
     */
    public Tensor weights() {
        return weights;
    }

    /**
     * Gets the bias of this layer.
     * @return the bias
     */
    public Tensor bias() {
        return bias;
    }

    /**
     * Gets the total number of biases in this layer.
     * @return 0 if bias is <code>null</code>, otherwise the number of elements in the bias tensor
     */
    public int totalBiases() {
        if (bias == null) return 0;

        return bias.elements();
    }

    /**
     * Gets the total number of weights in this layer.
     * @return 0 if the weights is <code>null</code>, otherwise the number of elements in the weights tensor
     */
    public int totalWeights() {
        if (weights == null) return 0;

        return weights.elements();
    }
    
    public List<ProtoModel.Tensor.Builder> weightsList() {
        return List.of(
            SerializeUtils.serializeTensor("weight", weights),
            SerializeUtils.serializeTensor("bias", bias)
        );
    }
}
