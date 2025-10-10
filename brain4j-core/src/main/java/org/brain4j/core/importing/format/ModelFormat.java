package org.brain4j.core.importing.format;

import org.brain4j.core.model.Model;

import java.io.File;

public interface ModelFormat {
    Model deserialize(File file);

    void serialize(Model model, File file);
}
