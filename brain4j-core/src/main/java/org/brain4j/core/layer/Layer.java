package org.brain4j.core.layer;

import com.google.gson.JsonObject;
import org.brain4j.core.loss.LossFunction;
import org.brain4j.core.training.optimizer.Optimizer;
import org.brain4j.core.training.updater.Updater;
import org.brain4j.math.activation.Activation;
import org.brain4j.math.activation.impl.LinearActivation;
import org.brain4j.math.clipper.GradientClipper;
import org.brain4j.math.clipper.impl.HardClipper;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.gpu.device.Device;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.weightsinit.WeightInitialization;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.random.RandomGenerator;

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
    public void initWeights(RandomGenerator generator, int input, int output) {
        // No-op
    }

    /**
     * Performs a forward pass through this layer.
     *
     * @param cache the states cache for this forward pass
     * @param inputs the input tensors
     * @return the output tensors
     */
    public abstract Tensor[] forward(StatesCache cache, Tensor... inputs);

    /**
     * Computes the loss (the gradient) with respect to the loss function and launches the autograd.
     * This method should only be called for the last layer of the neural network.
     *
     * @param cache the state cache of this inference
     * @param labels the label tensors
     * @param outputs the output tensors
     * @param lossFunction the loss function of this model
     */
    public void computeLoss(
        StatesCache cache,
        Tensor[] labels,
        Tensor[] outputs,
        LossFunction lossFunction
    ) {
        Tensor[] preOutputs = cache.output(this);
        
        if (labels.length != outputs.length) {
            throw new IllegalArgumentException("Targets amount does not equal to output amount.");
        }
        
        for (int i = 0; i < outputs.length; i++) {
            Tensor output = outputs[i];
            Tensor target = labels[i];
            Tensor preOutput = preOutputs[i];
            
            if (!Arrays.equals(output.shape(), target.shape())) {
                throw new IllegalArgumentException("Output and target shapes do not match! Output: " +
                    Arrays.toString(output.shape()) + ", Target: " + Arrays.toString(target.shape()));
            }
            
            Tensor error = output.minus(target);
            Tensor derivatives = activation.derivative(preOutput);
            Tensor delta = lossFunction.delta(error, derivatives);
            
            preOutput.backward(delta);
        }
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
     * Returns the output size of this layer, i.e. the number of neurons.
     * @return the output size
     */
    public abstract int size();
    
    public Tensor forward(StatesCache cache, Tensor input) {
        return forward(cache, new Tensor[] { input })[0];
    }
    
    /**
     * Checks if the amount of inputs is greater than the maximum amount.
     * If so, throws an exception, otherwise will do nothing.
     * @param length the maximum amount of accepted inputs
     * @param inputs the input tensors
     */
    public void checkInputLength(int length, Tensor... inputs) {
        if (inputs.length == length) return;

        throw new IllegalArgumentException(
            String.format("Input length mismatch! Got %s for layer %s but expecting %s inputs",
                inputs.length, this.getClass().getSimpleName(), length)
        );
    }
    
    public void serialize(JsonObject object) {
        // No-op
    }
    
    public void deserialize(JsonObject object) {
        // No-op
    }
    
    public void loadWeights(Map<String, Tensor> mappedWeights) {
        if (mappedWeights.containsKey("weights")) {
            this.weights = mappedWeights.get("weights");
        }
        
        if (mappedWeights.containsKey("bias")) {
            this.bias = mappedWeights.get("bias");
        }
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
     * Validates if the input can be passed as an input to this layer.
     * This is done by checking the input dimension and comparing it
     * to the layer's expected dimension.
     *
     * @param input the input tensor
     * @return <code>true</code> if the input is valid, <code>false</code> otherwise
     */
    public boolean validInput(Tensor input) {
        return true;
    }

    public Activation setActivation() {
        return activation;
    }

    public Layer setActivation(Activation activation) {
        this.activation = activation;
        return this;
    }

    public GradientClipper setClipper() {
        return clipper;
    }
    
    public Layer setClipper(GradientClipper clipper) {
        this.clipper = clipper;
        return this;
    }
    
    public WeightInitialization setWeightInit() {
        return weightInit;
    }

    public Layer setWeightInit(WeightInitialization weightInit) {
        this.weightInit = weightInit;
        return this;
    }

    public Tensor weights() {
        return weights;
    }
    
    public Layer setWeights(Tensor weights) {
        this.weights = weights;
        return this;
    }

    public Tensor bias() {
        return bias;
    }
    
    public Layer setBias(Tensor bias) {
        this.bias = bias;
        return this;
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
    
    public Map<String, Tensor> weightsMap() {
        Map<String, Tensor> result = new HashMap<>();
        
        if (weights != null) result.put("weights", weights);
        if (bias != null) result.put("bias", bias);
        
        return result;
    }
}
