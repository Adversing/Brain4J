package org.brain4j.core.training;

import org.brain4j.core.model.Model;
import org.brain4j.core.training.optimizer.Optimizer;
import org.brain4j.core.training.updater.Updater;
import org.brain4j.math.commons.Batch;
import org.brain4j.math.data.ListDataSource;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.gpu.GpuContext;
import org.brain4j.math.gpu.device.Device;
import org.brain4j.math.tensor.Tensor;

import java.util.function.BiConsumer;

public record BackPropagation(Model model, Optimizer optimizer, Updater updater) {

    public void propagatePartition(Batch batch, BiConsumer<Integer, Double> postBatchCallback, int index) {
        Device device = model.getDevice();
        StatesCache cache = new StatesCache(true, device);
        
        long start = System.nanoTime();
        
        Tensor[] inputs = batch.getFirst();
        Tensor[] labels = batch.getSecond();

        Tensor[] output = model.predict(cache, inputs);
        model.backpropagate(cache, output, labels);

        int elements = 0;
        
        for (Tensor input : inputs) {
            elements += input.shape()[0];
        }
        
        optimizer.postBatch();
        updater.postBatch(optimizer.getLearningRate(), elements);
        model.zeroGrad();
        
        if (device != null) {
            GpuContext.finishAndRelease(device);
        }
        
        double took = (System.nanoTime() - start) / 1e6;
        postBatchCallback.accept(index, took);
    }
    
    public void iteration(ListDataSource dataSource, BiConsumer<Integer, Double> postBatchCallback) {
        dataSource.reset();
        
        while (dataSource.hasNext()) {
            Batch batch = hostTo(dataSource.nextBatch());
            propagatePartition(batch, postBatchCallback, dataSource.getCursor());
        }
        
        updater.postFit(optimizer.getLearningRate(), dataSource.getSize());
        model.zeroGrad();
    }
    
    private Batch hostTo(Batch partition) {
        Device device = model.getDevice();
        
        Tensor[] inputs = partition.getFirst();
        Tensor[] labels = partition.getSecond();
        
        for (int i = 0; i < inputs.length; i++) {
            Tensor input = inputs[i];
            inputs[i] = input.to(device);
        }

        for (int i = 0; i < labels.length; i++) {
            Tensor label = labels[i];
            labels[i] = label.to(device);
        }
        
        return new Batch(inputs, labels);
    }
}