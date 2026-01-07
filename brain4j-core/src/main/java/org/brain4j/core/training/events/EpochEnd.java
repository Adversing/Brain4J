package org.brain4j.core.training.events;

import org.brain4j.core.training.impl.DefaultTrainer;

public record EpochEnd(DefaultTrainer trainer, int epoch, int totalEpochs) implements TrainingEvent {}
