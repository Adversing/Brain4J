package org.brain4j.core.importing.format;

import org.brain4j.core.model.Model;

import java.io.File;

public interface BinaryAdapter {
    Model deserialize(File file);
    void serialize(Model input, File file);
}
