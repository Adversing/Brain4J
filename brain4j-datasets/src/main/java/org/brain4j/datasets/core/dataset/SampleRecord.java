package org.brain4j.datasets.core.dataset;

import org.apache.avro.generic.GenericRecord;

public record SampleRecord(GenericRecord record) {
    
    public void put(String key, Object value) {
        record.put(key, value);
    }
    
    public Object get(String key) {
        return record.get(key);
    }
    
    public boolean hasField(String key) {
        return record.hasField(key);
    }
}
