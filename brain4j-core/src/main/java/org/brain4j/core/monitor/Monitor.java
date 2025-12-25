package org.brain4j.core.monitor;

import org.brain4j.core.training.events.TrainingEvent;

public interface Monitor {
    void onEvent(TrainingEvent event);
}
