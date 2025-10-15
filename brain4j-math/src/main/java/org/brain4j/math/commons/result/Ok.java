package org.brain4j.math.commons.result;

public record Ok<T, E extends Exception>(T value) implements Result<T, E> {
    @Override public boolean isOk() { return true; }
    @Override public boolean isErr() { return false; }
    @Override public T unwrap() { return value; }
    @Override public T unwrapOr(T defaultValue) { return value; }
    @Override public E error() { return null; }
}
