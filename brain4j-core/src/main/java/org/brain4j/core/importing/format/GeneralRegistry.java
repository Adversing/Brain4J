package org.brain4j.core.importing.format;

import org.brain4j.math.Commons;

import java.util.*;
import java.util.function.Function;
import java.util.function.Supplier;

public class GeneralRegistry<T, A> {
    
    private final Map<String, Class<? extends T>> idToClass = new HashMap<>();
    private final Map<Class<? extends T>, Set<String>> classToIds = new HashMap<>();
    private final Map<String, Function<A, T>> idToInstance = new HashMap<>();
    
    public void clear() {
        idToClass.clear();
        classToIds.clear();
        idToInstance.clear();
    }
    
    public void register(String identifier, Function<A, T> supplier) {
        idToInstance.put(identifier, supplier);
    }
    
    public void register(String identifier, Class<? extends T> clazz) {
        idToClass.put(identifier, clazz);
        classToIds.computeIfAbsent(clazz, k -> new HashSet<>()).add(identifier);
    }
    
    public String fromClass(Class<? extends T> clazz) {
        Set<String> ids = classToIds.get(clazz);
        return (ids == null || ids.isEmpty()) ? null : ids.iterator().next();
    }
    
    public Set<String> aliasesFromClass(Class<? extends T> clazz) {
        return classToIds.getOrDefault(clazz, Collections.emptySet());
    }
    
    public Class<? extends T> fromId(String identifier) {
        return idToClass.get(identifier);
    }

    public T toInstance(String identifier) {
        return toInstance(identifier, null);
    }

    public T toInstance(String identifier, A value) {
        if (idToInstance.containsKey(identifier)) {
            return idToInstance.get(identifier).apply(value);
        }

        return Commons.newInstance(fromId(identifier));
    }
}
