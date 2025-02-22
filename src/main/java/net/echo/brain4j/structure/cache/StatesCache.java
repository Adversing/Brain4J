package net.echo.brain4j.structure.cache;

import net.echo.brain4j.structure.Neuron;
import net.echo.brain4j.structure.Synapse;

public class StatesCache {

    private final double[] gradients;
    private final double[] valuesCache;
    private final double[] deltasCache;

    public StatesCache() {
        this.gradients = new double[Parameters.TOTAL_SYNAPSES];
        this.valuesCache = new double[Parameters.TOTAL_NEURONS];
        this.deltasCache = new double[Parameters.TOTAL_NEURONS];
    }

    public double[] getGradients() {
        return gradients;
    }

    public double getGradient(int index) {
        return gradients[index];
    }

    public double getValue(Neuron neuron) {
        return valuesCache[neuron.getId()];
    }

    public double getDelta(Neuron neuron) {
        return this.deltasCache[neuron.getId()];
    }

    public void setValue(Neuron neuron, double value) {
        this.valuesCache[neuron.getId()] = value;
    }

    public void setDelta(Neuron neuron, double delta) {
        this.deltasCache[neuron.getId()] = delta;
    }

    public void setGradient(int index, double gradient) {
        this.gradients[index] = gradient;
    }

    public void addDelta(Neuron neuron, double delta) {
        this.deltasCache[neuron.getId()] += delta;
    }

    public void addGradient(int index, double change) {
        this.gradients[index] += change;
    }
}
