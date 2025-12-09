package org.brain4j.core.training.wrappers;

import org.brain4j.math.data.ListDataSource;

/**
 * Defines the parameters used to train and validate the model.
 * <p>If {@code epochs} is not defined, a default value of 1 is used instead.</p>
 * <p>If {@code evaluateEvery} is not defined, a default value of {@code Integer.MAX_VALUE} is used instead.</p>
 *
 * @param trainSet the training dataset, used for backpropagation
 * @param validationSet the validation dataset, used for performance evaluation
 * @param epochs the number of epochs to fit the model on
 * @param evaluateEvery how often (in epochs) the model should be evaluated
 */
public record TrainingParams(
    ListDataSource trainSet,
    ListDataSource validationSet,
    int epochs,
    int evaluateEvery
) {
    public TrainingParams {
        if (trainSet == null) throw new NullPointerException("Training set must not be null!");
        if (validationSet == null) throw new NullPointerException("Validation set must not be null!");
        if (epochs <= 0) throw new IllegalArgumentException("Number of epochs must be greater than zero!");
        if (evaluateEvery <= 0) throw new IllegalArgumentException("Number of evaluate every must be greater than zero!");
    }

    public TrainingParams(ListDataSource trainSource) {
        this(trainSource, trainSource, 1, Integer.MAX_VALUE);
    }

    public TrainingParams(ListDataSource trainSource, ListDataSource validationSource) {
        this(trainSource, validationSource, 1, Integer.MAX_VALUE);
    }

    public TrainingParams(int epochs, ListDataSource validationSet, ListDataSource trainSet) {
        this(trainSet, validationSet, epochs, Integer.MAX_VALUE);
    }
}
