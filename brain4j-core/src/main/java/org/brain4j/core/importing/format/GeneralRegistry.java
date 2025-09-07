package org.brain4j.core.importing.format;

import java.util.HashMap;
import java.util.Map;

public class GeneralRegistry<T> {
    
    private final Map<String, Class<? extends T>> idToClass = new HashMap<>();
    private final Map<Class<? extends T>, String> classToId = new HashMap<>();

    public void clear() {
        idToClass.clear();
        classToId.clear();
    }

    public void register(String identifier, Class<? extends T> clazz) {
        idToClass.put(identifier, clazz);
        classToId.put(clazz, identifier);
    }

    public String fromClass(Class<? extends T> clazz) {
        return classToId.get(clazz);
    }
    
    public Class<? extends T> fromId(String identifier) {
        return idToClass.get(identifier);
    }
}
