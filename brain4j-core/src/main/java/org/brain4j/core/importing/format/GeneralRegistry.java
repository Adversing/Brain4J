package org.brain4j.core.importing.format;

import org.brain4j.math.Commons;

import java.util.*;
import java.util.function.Supplier;

public class GeneralRegistry<T> {
    
    private final Map<String, Class<? extends T>> idToClass = new HashMap<>();
    private final Map<Class<? extends T>, Set<String>> classToIds = new HashMap<>();
    private final Map<String, Supplier<T>> idToInstance = new HashMap<>();
    
    public void clear() {
        idToClass.clear();
        classToIds.clear();
        idToInstance.clear();
    }
    
    public void register(String identifier, Supplier<T> supplier) {
        T instance = supplier.get();
        Class<? extends T> clazz = (Class<? extends T>) instance.getClass();
        
        idToClass.put(identifier, clazz);
        classToIds.computeIfAbsent(clazz, k -> new HashSet<>()).add(identifier);
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
        if (idToInstance.containsKey(identifier)) {
            return idToInstance.get(identifier).get();
        }
        
        return Commons.newInstance(fromId(identifier));
    }
}
