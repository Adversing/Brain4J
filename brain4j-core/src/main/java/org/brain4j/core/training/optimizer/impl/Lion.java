package org.brain4j.core.training.optimizer.impl;

import org.brain4j.core.training.optimizer.Optimizer;
import org.brain4j.math.Tensors;
import org.brain4j.math.tensor.Tensor;

import java.util.HashMap;
import java.util.Map;

public class Lion extends Optimizer {

    protected Map<Tensor, Tensor> momentumHistory = new HashMap<>();
    protected double beta1 = 0.9;
    protected double beta2 = 0.99;
    
    protected Lion() {
        super(0);
    }

    public Lion(double learningRate) {
        super(learningRate);
    }

    public Lion(double learningRate, double beta, double beta2) {
        super(learningRate);
        this.beta1 = beta;
        this.beta2 = beta2;
    }

    @Override
    public Tensor step(Tensor weights, Tensor gradient) {
        Tensor scaledGrad = gradient.mul(1 - beta2);

        Tensor newMomentum = calcMomentum(weights, gradient);
        Tensor scaledMomentum = newMomentum.times(beta2);
        Tensor update = scaledMomentum.add(scaledGrad.broadcastLike(scaledMomentum)).sign();

        momentumHistory.put(weights, newMomentum);
        return update;
    }

    public Tensor calcMomentum(Tensor weights, Tensor gradient) {
        Tensor momentum = momentumHistory.getOrDefault(weights, Tensors.zerosLike(gradient));
        Tensor scaledGrad = gradient.mul(1 - beta1);
        return momentum.mul(beta1).add(scaledGrad.broadcastLike(momentum));
    }

    @Override
    public void initialize() {
        this.momentumHistory = new HashMap<>();
    }

    public Map<Tensor, Tensor> momentumHistory() {
        return momentumHistory;
    }

    public double beta1() {
        return beta1;
    }

    public Lion setBeta1(double beta1) {
        this.beta1 = beta1;
        return this;
    }

    public double beta2() {
        return beta2;
    }

    public Lion setBeta2(double beta2) {
        this.beta2 = beta2;
        return this;
    }
}