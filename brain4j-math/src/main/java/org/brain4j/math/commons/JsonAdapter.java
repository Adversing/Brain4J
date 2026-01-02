package org.brain4j.math.commons;

import com.google.gson.JsonObject;

public interface JsonAdapter {
    void serialize(JsonObject object);
    void deserialize(JsonObject object);
}
