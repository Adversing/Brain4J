package org.brain4j.core.training.events;

import org.brain4j.core.training.impl.DefaultTrainer;

public record EpochStart(DefaultTrainer trainer, int epoch, int totalEpochs) implements TrainingEvent {}
