package org.brain4j.core.training.events;

public sealed interface TrainingEvent permits BatchEnd, BatchStart, EpochEnd, EpochStart, TrainingEnd {}
