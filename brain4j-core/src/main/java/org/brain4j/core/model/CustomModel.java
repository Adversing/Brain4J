package org.brain4j.core.model;

import org.brain4j.math.Pair;
import org.brain4j.math.Tensors;
import org.brain4j.math.data.ListDataSource;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.index.Range;
import org.brain4j.core.layer.Layer;
import org.brain4j.core.loss.LossFunction;
import org.brain4j.core.loss.impl.BinaryCrossEntropy;
import org.brain4j.core.training.StatesCache;
import org.brain4j.core.training.optimizer.Optimizer;
import org.brain4j.core.training.updater.Updater;
import org.brain4j.core.training.wrappers.EvaluationResult;

import java.util.*;
import java.util.concurrent.atomic.AtomicReference;
import java.util.stream.IntStream;

public abstract class CustomModel {
    
    protected Optimizer optimizer;
    protected Updater updater;
    protected LossFunction lossFunction;
    
    public CustomModel(Optimizer optimizer, Updater updater, LossFunction lossFunction) {
        this.optimizer = optimizer;
        this.updater = updater;
        this.lossFunction = lossFunction;
    }

    public abstract Tensor[] forward(StatesCache cache, Tensor... inputs);
    
    public abstract void fit(StatesCache cache, Tensor[] output, Tensor[] label);
    
    public abstract void zeroGrad();
    
    public Optimizer optimizer() {
        return optimizer;
    }
    
    public Updater updater() {
        return updater;
    }
    
    public LossFunction lossFunction() {
        return lossFunction;
    }
    
    public long seed() {
        return System.currentTimeMillis();
    }
    
    public EvaluationResult evaluate(ListDataSource dataSource) {
        int classes = Math.max(2, dataSource.samples().getFirst().label().elements());
        Map<Integer, Tensor> classifications = new HashMap<>();
        
        for (int i = 0; i < classes; i++) {
            classifications.put(i, Tensors.zeros(classes));
        }
        
        AtomicReference<Double> totalLoss = new AtomicReference<>(0.0);
        
        dataSource.reset();
        
        while (dataSource.hasNext()) {
            Pair<Tensor[], Tensor[]> batch = dataSource.nextBatch();
            makeEvaluation(batch, classifications, totalLoss);
        }
        
        return new EvaluationResult(totalLoss.get() / dataSource.size(), classes, classifications);
    }
    
    public void updateWeights(Tensor[] outputs) {
        int elements = 0;
        
        for (Tensor output : outputs) {
            elements += output.shape(0);
        }
        
        updater.updateWeights(optimizer.learningRate(), elements);
    }
    
    protected void makeEvaluation(
        Pair<Tensor[], Tensor[]> batch,
        Map<Integer, Tensor> classifications,
        AtomicReference<Double> totalLoss
    ) {
        Tensor[] inputs = batch.first();
        Tensor[] labels = batch.second();

        Tensor[] outputs = forward(new StatesCache(false), inputs);

        for (Tensor input : inputs) {
            int batchSize = input.shape(0);

            for (int i = 0; i < outputs.length; i++) {
                Tensor output = outputs[i];
                Tensor label = labels[i];

                for (int b = 0; b < batchSize; b++) {
                    Range range = Range.point(b);

                    Tensor sampleOutput = output.slice(range).flatten();
                    Tensor sampleLabel = label.slice(range).flatten();

                    int predIndex = sampleOutput.argmax();
                    int targetIndex = sampleLabel.argmax();

                    if (sampleOutput.elements() == 1 && lossFunction instanceof BinaryCrossEntropy) {
                        predIndex = sampleOutput.get(0) > 0.5 ? 1 : 0;
                        targetIndex = (int) sampleLabel.get(0);
                    }

                    double loss = lossFunction.calculate(sampleLabel, sampleOutput);
                    totalLoss.updateAndGet(v -> v + loss);

                    Tensor predictions = classifications.get(targetIndex);
                    int pred = (int) predictions.get(predIndex);

                    predictions.set(pred + 1, predIndex);
                }
            }
        }
    }
    
    protected final void connect(Layer... layers) {
        optimizer.initialize();
        updater.resetGradients();
        
        if (layers.length == 0) return;
        
        Layer previous = null;
        int size = layers.length;
        
        for (Layer layer : layers) {
            previous = layer.connect(previous);
        }
        
        int[] inputSizes = new int[size];
        
        for (int i = 1; i < inputSizes.length; i++) {
            inputSizes[i] = layers[i - 1].size();
        }
        
        IntStream.range(0, size).parallel().forEach(i -> {
            Layer layer = layers[i];
            
            int input = inputSizes[i];
            int output = layer.size();
            
            Random localRandom = Random.from(new SplittableRandom(seed() + i));
            layer.initWeights(localRandom, input, output);
        });
    }
}
