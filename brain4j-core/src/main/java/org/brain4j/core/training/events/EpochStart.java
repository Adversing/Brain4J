package org.brain4j.core.training.events;

import org.brain4j.core.training.Trainer;

public record EpochStart(Trainer trainer, int epoch, int total) implements TrainingEvent {}
