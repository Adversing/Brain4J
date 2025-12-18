package org.brain4j.math.commons.result;

/**
 * A type that represents either success (Ok) or failure (Err).
 *
 * <p>This interface provides a way to handle computation that may fail, similar to
 * Rust's Result type. It is particularly useful for operations that can fail with
 * a specific exception type rather than using runtime exceptions.
 *
 * <p>Example usage:
 * <pre>{@code
 * Result<Integer, IOException> result = computeSomething();
 * if (result.isOk()) {
 *     Integer value = result.unwrap();
 *     // use value
 * } else {
 *     IOException error = result.error();
 *     // handle error
 * }
 * }</pre>
 *
 * @param <T> the type of the success value
 * @param <E> the type of error/exception
 */
@Deprecated(forRemoval = true)
public sealed interface Result<T, E extends Exception> permits Ok, Err {

    /**
     * Returns true if this is an Ok variant.
     *
     * @return true if this Result contains a success value
     */
    boolean isOk();

    /**
     * Returns true if this is an Err variant.
     *
     * @return true if this Result contains an error
     */
    boolean isErr();

    /**
     * Extracts the success value if this is Ok, otherwise throws the error.
     *
     * @return the contained Ok value
     * @throws E if this is an Err variant
     */
    T unwrap() throws E;

    /**
     * Returns the success value or a provided default.
     *
     * @param defaultValue the value to return if this is an Err
     * @return the contained Ok value or the provided default
     */
    T unwrapOr(T defaultValue);

    /**
     * Returns the contained error.
     * 
     * @return the error if this is Err, or null if this is Ok
     */
    E error();

    /**
     * Creates a new Ok Result containing the given value.
     *
     * @param value the success value
     * @param <T> type of the success value
     * @param <E> type of possible errors
     * @return a new Ok Result
     */
    static <T, E extends Exception> Result<T, E> ok(T value) {
        return new Ok<>(value);
    }

    /**
     * Creates a new Err Result containing the given error.
     *
     * @param error the error value
     * @param <T> type of possible success values
     * @param <E> type of the error
     * @return a new Err Result
     */
    static <T, E extends Exception> Result<T, E> err(E error) {
        return new Err<>(error);
    }
}