package org.brain4j.core.training;

import org.brain4j.core.layer.Layer;
import org.brain4j.core.loss.LossFunction;
import org.brain4j.core.model.Model;
import org.brain4j.core.monitor.Monitor;
import org.brain4j.core.training.optimizer.Optimizer;
import org.brain4j.core.training.updater.Updater;
import org.brain4j.math.commons.Batch;
import org.brain4j.math.data.ListDataSource;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.tensor.Tensor;

import java.util.List;

public record Trainer(Model model, Monitor monitor, TrainingConfig config) {

    public static Trainer compile(Model model, Monitor monitor, LossFunction loss, Optimizer optimizer, Updater updater) {
        return new Trainer(model, monitor, new TrainingConfig(loss, optimizer, updater));
    }

    public void fit(ListDataSource dataSource, int epochs) {
        for (int i = 0; i < epochs; i++) {
            fit(dataSource);
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
        Batch batch = dataSource.nextBatch();

        Tensor[] inputs = batch.getFirst();
        Tensor[] targets = batch.getSecond();

        StatesCache cache = StatesCache.withTraining();
        Tensor[] outputs = model.predict(cache, batch.getFirst());

        Optimizer optimizer = config.optimizer();
        LossFunction loss = config.loss();
        Updater updater = config.updater();

        List<Layer> layers = model.getFlattened();

        layers.getLast().computeLoss(cache, targets, outputs, loss);
        layers.forEach(layer -> layer.backward(cache, updater, optimizer));

        int elements = 0;

        for (Tensor input : inputs) elements += input.shape(0);

        optimizer.postBatch();
        updater.postBatch(optimizer.getLearningRate(), elements);

        layers.forEach(Layer::resetGrad);
    }
}
