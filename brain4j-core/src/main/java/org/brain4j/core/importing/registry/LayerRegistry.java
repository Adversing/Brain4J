package org.brain4j.core.importing.registry;

import org.brain4j.core.layer.Layer;

import java.lang.reflect.Constructor;
import java.util.HashMap;
import java.util.Map;

public class LayerRegistry {

    private final Map<String, Class<? extends Layer>> registry = new HashMap<>();

    public void clear() {
        registry.clear();
    }

    public void register(String identifier, Class<? extends Layer> clazz) {
        registry.put(identifier, clazz);
    }

    public void registerAll(Map<String, Class<? extends Layer>> map) {
        registry.putAll(map);
    }

    public Class<? extends Layer> get(String identifier) {
        return registry.get(identifier);
    }

    public <T extends Layer> T create(String identifier) {
        Class<? extends Layer> clazz = registry.get(identifier);

        try {
            Constructor<?> constructor = clazz.getDeclaredConstructor();
            constructor.setAccessible(true);

            return (T) constructor.newInstance();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    public String fromState(Class<? extends Layer> clazz) {
        for (Map.Entry<String, Class<? extends Layer>> entry : registry.entrySet()) {
            String identifier = entry.getKey();
            Class<? extends Layer> layer = entry.getValue();

            if (layer == clazz) {
                return identifier;
            }
        }

        return null;
    }
}
