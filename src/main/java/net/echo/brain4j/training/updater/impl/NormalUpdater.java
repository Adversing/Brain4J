package net.echo.brain4j.training.updater.impl;

import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.model.Model;
import net.echo.brain4j.structure.Neuron;
import net.echo.brain4j.structure.cache.Parameters;
import net.echo.brain4j.structure.cache.StatesCache;
import net.echo.brain4j.structure.Synapse;
import net.echo.brain4j.training.updater.Updater;

public class NormalUpdater extends Updater {

    protected Synapse[] synapses;
    protected double[] gradients;

    @Override
    public void postInitialize(Model model) {
        this.synapses = new Synapse[Parameters.TOTAL_SYNAPSES];
        this.gradients = new double[Parameters.TOTAL_SYNAPSES];

        for (Layer layer : model.getLayers()) {
            for (Synapse synapse : layer.getSynapses()) {
                synapses[synapse.getSynapseId()] = synapse;
            }
        }
    }

    @Override
    public void postFit(Model model, double learningRate) {
        for (int i = 0; i < gradients.length; i++) {
            Synapse synapse = synapses[i];
            double gradient = gradients[i];

            // FOR FUTURE ECHO: DO NOT TOUCH THIS!!!!! MULTIPLYING FOR THE LEARNING RATE IS IMPORTANT AND IDK WHY
            synapse.setWeight(synapse.getWeight() - learningRate * gradient);
        }

        model.reloadMatrices();

        for (Layer layer : model.getLayers()) {
            for (Neuron neuron : layer.getNeurons()) {
                double deltaBias = learningRate * neuron.getTotalDelta();

                neuron.setBias(neuron.getBias() - deltaBias);
                neuron.setTotalDelta(0.0);
            }
        }

        this.gradients = new double[Parameters.TOTAL_SYNAPSES];
    }

    @Override
    public void acknowledgeChange(StatesCache cacheHolder, Synapse synapse, double change) {
        this.gradients[synapse.getSynapseId()] += change;
    }
}
