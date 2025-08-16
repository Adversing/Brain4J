package org.brain4j.core.training;

import org.brain4j.common.Pair;
import org.brain4j.common.data.ListDataSource;
import org.brain4j.common.gpu.GpuContext;
import org.brain4j.common.gpu.device.Device;
import org.brain4j.common.tensor.Tensor;
import org.brain4j.core.model.Model;
import org.brain4j.core.training.optimizer.Optimizer;
import org.brain4j.core.training.updater.Updater;

import java.util.function.BiConsumer;

public record BackPropagation(Model model, Optimizer optimizer, Updater updater) {
    
    public BackPropagation(Model model, Optimizer optimizer, Updater updater) {
        this.model = model;
        this.optimizer = optimizer;
        this.updater = updater;
        updater.resetGradients();
    }
    
    public void propagatePartition(
        Pair<Tensor[], Tensor> batch,
        BiConsumer<Integer, Double> postBatchCallback,
        int index
    ) {
        Device device = model.device();
        StatesCache cache = new StatesCache(device);
        
        long start = System.nanoTime();
        
        Tensor[] inputs = batch.first();
        Tensor labels = batch.second();
        
        Tensor output = model.predict(cache, true, inputs);
        model.backpropagate(cache, output, labels);
        
        int elements = 1;
        
        for (Tensor input : inputs) {
            elements *= input.shape()[0];
        }
        
        optimizer.postBatch();
        updater.postBatch(elements);
        model.zeroGrad();
        
        if (device != null) {
            GpuContext.closeQueue(device);
        }
        
        double took = (System.nanoTime() - start) / 1e6;
        postBatchCallback.accept(index, took);
    }
    
    public void iteration(ListDataSource dataSource, BiConsumer<Integer, Double> postBatchCallback) {
        dataSource.reset();
        
        while (dataSource.hasNext()) {
            Pair<Tensor[], Tensor> batch = hostTo(dataSource.nextBatch());
            propagatePartition(batch, postBatchCallback, dataSource.cursor());
        }
        
        updater.postFit(dataSource.size());
        model.zeroGrad();
    }
    
    private Pair<Tensor[], Tensor> hostTo(Pair<Tensor[], Tensor> partition) {
        Device device = model.device();
        
        Tensor[] inputs = partition.first();
        Tensor labels = partition.second().to(device);
        
        for (int i = 0; i < inputs.length; i++) {
            Tensor input = inputs[i];
            inputs[i] = input.to(device);
        }
        
        return new Pair<>(inputs, labels);
    }
}