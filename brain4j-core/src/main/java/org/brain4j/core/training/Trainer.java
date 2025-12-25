package org.brain4j.core.training;

import org.brain4j.core.layer.Layer;
import org.brain4j.core.loss.LossFunction;
import org.brain4j.core.model.Model;
import org.brain4j.core.monitor.Monitor;
import org.brain4j.core.training.events.BatchEnd;
import org.brain4j.core.training.events.BatchStart;
import org.brain4j.core.training.events.EpochEnd;
import org.brain4j.core.training.events.EpochStart;
import org.brain4j.core.training.optimizer.Optimizer;
import org.brain4j.core.training.updater.Updater;
import org.brain4j.math.commons.Batch;
import org.brain4j.math.data.ListDataSource;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.tensor.Tensor;

import java.util.List;

public record Trainer(Model model, List<Monitor> monitors, TrainingConfig config) {

    public static Trainer compile(Model model, List<Monitor> monitors, LossFunction loss, Optimizer optimizer, Updater updater) {
        return new Trainer(model, monitors, new TrainingConfig(loss, optimizer, updater));
    }

    public Trainer {
        if (model == null) throw new IllegalArgumentException("Model cannot be null!");
        if (config == null) throw new IllegalArgumentException("Config cannot be null!");
        
        config.optimizer().initialize();
        config.updater().initialize();
    }
    
    public void fit(ListDataSource dataSource, int epochs) {
        for (int i = 0; i < epochs; i++) {
            EpochStart start = new EpochStart(this, i, epochs);
            EpochEnd end = new EpochEnd(this, i, epochs);
            
            monitors.forEach(x -> x.onEvent(start));
            fit(dataSource);
            monitors.forEach(x -> x.onEvent(end));
        }
    }

    public void fit(ListDataSource dataSource) {
        dataSource.reset();

        while (dataSource.hasNext()) {
            fitBatch(dataSource);
        }

        Optimizer optimizer = config.optimizer();
        Updater updater = config.updater();

        updater.postFit(optimizer.getLearningRate(), dataSource.getSize());
    }

    public void fitBatch(ListDataSource dataSource) {
        int current = dataSource.getCursor(), total = dataSource.getBatches();
        Batch batch = dataSource.nextBatch();

        Tensor[] inputs = batch.getFirst();
        Tensor[] targets = batch.getSecond();
        
        BatchStart start = new BatchStart(this, current, total);
        BatchEnd end = new BatchEnd(this, current, total);
        
        monitors.forEach(x -> x.onEvent(start));
        
        StatesCache cache = StatesCache.withTraining();
        Tensor[] outputs = model.predict(cache, batch.getFirst());

        Optimizer optimizer = config.optimizer();
        LossFunction loss = config.loss();
        Updater updater = config.updater();

        List<Layer> layers = model.getLayers();

        layers.getLast().computeLoss(cache, targets, outputs, loss);
        layers.forEach(layer -> layer.backward(cache, updater, optimizer));

        int elements = 0;

        for (Tensor input : inputs) elements += input.shape(0);

        optimizer.postBatch();
        updater.postBatch(optimizer.getLearningRate(), elements);

        layers.forEach(Layer::resetGrad);
        monitors.forEach(x -> x.onEvent(end));
    }
}
