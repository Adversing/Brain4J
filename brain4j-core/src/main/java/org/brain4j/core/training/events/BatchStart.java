package org.brain4j.core.training.events;

import org.brain4j.core.training.impl.DefaultTrainer;

public record BatchStart(DefaultTrainer trainer, int batch, int totalBatches) implements TrainingEvent {}
