package org.brain4j.core.model.impl;

import org.brain4j.common.Pair;
import org.brain4j.common.Tensors;
import org.brain4j.common.data.ListDataSource;
import org.brain4j.common.tensor.Tensor;
import org.brain4j.common.tensor.index.Range;
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
    
    public abstract Tensor forward(StatesCache cache, Tensor... inputs);
    
    public abstract void fit(StatesCache cache, Tensor output, Tensor label);
    
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
            Pair<Tensor[], Tensor> batch = dataSource.nextBatch();
            makeEvaluation(batch, classifications, totalLoss);
        }
        
        return new EvaluationResult(totalLoss.get() / dataSource.size(), classes, classifications);
    }
    
    public void updateWeights(Tensor output) {
        int elements = output.shape()[0];
        updater.updateWeights(elements);
    }
    
    protected void makeEvaluation(
        Pair<Tensor[], Tensor> batch,
        Map<Integer, Tensor> classifications,
        AtomicReference<Double> totalLoss
    ) {
        Tensor[] inputs = batch.first(); // [batch_size, input_size]
        Tensor expected = batch.second(); // [batch_size, output_size]
        
        Tensor prediction = forward(new StatesCache(), inputs).cpu(); // [batch_size, output_size]
        
        for (Tensor input : inputs) {
            int batchSize = input.shape()[0];
            
            for (int i = 0; i < batchSize; i++) {
                Range range = Range.point(i);
                
                Tensor output = prediction.slice(range).flatten();
                Tensor target = expected.slice(range).flatten();
                
                int predIndex = output.argmax();
                int targetIndex = target.argmax();
                
                if (output.elements() == 1 && lossFunction instanceof BinaryCrossEntropy) {
                    predIndex = output.get(0) > 0.5 ? 1 : 0;
                    targetIndex = (int) target.get(0);
                }
                
                double loss = lossFunction.calculate(target, output);
                totalLoss.updateAndGet(v -> v + loss);
                
                Tensor predictions = classifications.get(targetIndex);
                int pred = (int) predictions.get(predIndex);
                
                predictions.set(pred + 1, predIndex);
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
