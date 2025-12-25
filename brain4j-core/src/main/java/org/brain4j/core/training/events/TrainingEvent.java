package org.brain4j.core.training.events;

public sealed interface TrainingEvent permits BatchStart, BatchEnd, EpochStart, EpochEnd {}
