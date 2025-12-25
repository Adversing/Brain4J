package org.brain4j.core.training;

import org.brain4j.core.loss.LossFunction;
import org.brain4j.core.training.optimizer.Optimizer;
import org.brain4j.core.training.updater.Updater;

public record TrainingConfig(LossFunction loss, Optimizer optimizer, Updater updater) {

    public static Builder builder() {
        return new TrainingConfig.Builder();
    }

    public static class Builder {
        private LossFunction loss;
        private Optimizer optimizer;
        private Updater updater;

        public Builder setLoss(LossFunction loss) {
            this.loss = loss;
            return this;
        }

        public Builder setOptimizer(Optimizer optimizer) {
            this.optimizer = optimizer;
            return this;
        }

        public Builder setUpdater(Updater updater) {
            this.updater = updater;
            return this;
        }

        public TrainingConfig build() {
            if (loss == null) throw new IllegalStateException("Loss function has not been set");
            if (optimizer == null) throw new IllegalStateException("Optimizer has not been set");
            if (updater == null) throw new IllegalStateException("Updater has not been set");
            return new TrainingConfig(loss, optimizer, updater);
        }
    }
}
