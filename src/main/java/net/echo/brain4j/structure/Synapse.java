package net.echo.brain4j.structure;

public class Synapse {

    public static int ID_COUNTER;

    private final Neuron inputNeuron;
    private final Neuron outputNeuron;
    private final int synapseId;
    private double weight;

    public Synapse(Neuron inputNeuron, Neuron outputNeuron, double bound) {
        this.synapseId = ID_COUNTER++;
        this.inputNeuron = inputNeuron;
        this.outputNeuron = outputNeuron;
        this.weight = (Math.random() * 2 * bound) - bound;
    }

    public Neuron getInputNeuron() {
        return inputNeuron;
    }

    public Neuron getOutputNeuron() {
        return outputNeuron;
    }

    public int getSynapseId() {
        return synapseId;
    }

    public double getWeight() {
        return weight;
    }

    public void setWeight(double weight) {
        this.weight = weight;
    }
}
