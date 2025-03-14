package net.echo.brain4j.transformers.attention;

import com.google.common.base.Preconditions;
import net.echo.brain4j.model.initialization.WeightInit;
import net.echo.brain4j.utils.Vector;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class MultiHeadAttention {

    private final List<AttentionHead> heads;
    protected final WeightInit weightInit;
    protected final double temperature;
    protected final int headCount;
    protected final int modelDimension;
    protected final int headDimension;

    protected final float[][] outProjectionWeights;

    public MultiHeadAttention(WeightInit weightInit, int headCount, int modelDimension, double temperature) {
        this.weightInit = weightInit;
        this.headCount = headCount;
        this.modelDimension = modelDimension;
        this.temperature = temperature;

        Preconditions.checkState(modelDimension % headCount == 0, "Model dimension must be divisible by head count!");

        this.headDimension = modelDimension / headCount;
        this.heads = new ArrayList<>();
        this.outProjectionWeights = new float[headCount * headDimension][modelDimension];

        initializeHeads();
        initializeOutProjectionWeights();
    }

    public List<Vector> attend(List<Vector> inputs) {
        List<List<Vector>> headOutputs = new ArrayList<>();

        for (AttentionHead head : heads) {
            headOutputs.add(head.attend(inputs));
        }

        return concatenate(headOutputs, inputs);
    }

    public List<Vector> concatenate(List<List<Vector>> headOutputs, List<Vector> inputs) {
        List<Vector> result = new ArrayList<>();

        for (int i = 0; i < inputs.size(); i++) {
            List<Vector> concatList = new ArrayList<>();

            for (List<Vector> headOutput : headOutputs) {
                concatList.add(headOutput.get(i));
            }

            Vector concatenated = concatenateVectors(concatList);
            Vector projected = projectVector(concatenated);

            projected.add(inputs.get(i));
            result.add(projected);
        }

        return result;
    }

    public int getTotalNeurons() {
        int total = 0;

        total += outProjectionWeights.length * modelDimension;

        for (AttentionHead head : heads) {
            total += head.size();
        }

        return total;
    }

    protected void initializeHeads() {
        for (int i = 0; i < headCount; i++) {
            heads.add(new AttentionHead(weightInit, modelDimension, headDimension, temperature));
        }
    }

    protected void initializeOutProjectionWeights() {
        Random rng = new Random();
        double bound = weightInit.getInitializer().getBound(headCount * headDimension, modelDimension);

        for (int i = 0; i < headCount * headDimension; i++) {
            for (int j = 0; j < modelDimension; j++) {
                double value = (rng.nextDouble() * 2 * bound) - bound;
                outProjectionWeights[i][j] = (float) value;
            }
        }
    }

    protected Vector projectVector(Vector concatenated) {
        Vector result = new Vector(modelDimension);

        for (int j = 0; j < modelDimension; j++) {
            double sum = 0.0;

            for (int i = 0; i < concatenated.size(); i++) {
                sum += concatenated.get(i) * outProjectionWeights[i][j];
            }

            result.set(j, sum);
        }

        return result;
    }

    protected Vector concatenateVectors(List<Vector> vectors) {
        int totalSize = 0;

        for (Vector v : vectors) {
            totalSize += v.size();
        }

        Vector concatenated = new Vector(totalSize);
        int index = 0;

        for (Vector v : vectors) {
            for (int i = 0; i < v.size(); i++) {
                concatenated.set(index++, v.get(i));
            }
        }

        return concatenated;
    }
}
