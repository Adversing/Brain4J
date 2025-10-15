package org.brain4j.math.commons.result;

import java.util.function.Consumer;
import java.util.function.Function;

public sealed interface Result<T, E extends Exception> permits Ok, Err {

    boolean isOk();
    boolean isErr();

    T unwrap() throws E;
    T unwrapOr(T defaultValue);
    E error();

    static <T, E extends Exception> Result<T, E> ok(T value) {
        return new Ok<>(value);
    }

    static <T, E extends Exception> Result<T, E> err(E error) {
        return new Err<>(error);
    }
}