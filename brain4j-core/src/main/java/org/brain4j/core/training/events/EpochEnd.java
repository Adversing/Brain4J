package org.brain4j.core.training.events;

import org.brain4j.core.training.Trainer;

public record EpochEnd(Trainer trainer, int epoch, int totalEpochs) implements TrainingEvent {}
