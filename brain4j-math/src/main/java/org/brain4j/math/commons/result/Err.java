package org.brain4j.math.commons.result;

public record Err<T, E extends Exception>(E error) implements Result<T, E> {
    @Override public boolean isOk() { return false; }
    @Override public boolean isErr() { return true; }
    @Override public T unwrap() throws E { throw error; }
    @Override public T unwrapOr(T defaultValue) { return defaultValue; }
    @Override public E error() { return error; }
}
