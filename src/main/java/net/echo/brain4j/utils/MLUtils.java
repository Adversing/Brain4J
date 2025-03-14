package net.echo.brain4j.utils;

import java.util.List;

/**
 * Utility class for conversions and value matching.
 */
public class MLUtils {

    private static final double GRADIENT_CLIP = 10.0;

    /**
     * Finds the best matching enum constant based on output values.
     *
     * @param outputs array of output values
     * @param clazz   the enum class
     * @param <T>     the type of the enum
     * @return the best matching enum constant
     */
    public static <T extends Enum<T>> T parse(Vector outputs, Class<T> clazz) {
        return clazz.getEnumConstants()[indexOfMaxValue(outputs)];
    }

    /**
     * Gets the header with a middle text for display.
     *
     * @param middleText the middle text
     * @return the combined header
     */
    public static String getHeader(String middleText) {
        String base = "=";

        int maxLength = 70;
        int middleLength = middleText.length();

        String repeated = base.repeat((maxLength - middleLength) / 2);
        String result = repeated + middleText + repeated;

        return (result.length() != maxLength ? result + "=" : result) + "\n";
    }

    /**
     * Finds the index of the maximum value in an array.
     *
     * @param inputs array of input values
     * @return index of the maximum value
     */
    public static int indexOfMaxValue(Vector inputs) {
        int index = 0;
        double max = inputs.get(0);

        for (int i = 1; i < inputs.size(); i++) {
            if (inputs.get(i) > max) {
                max = inputs.get(i);
                index = i;
            }
        }

        return index;
    }

    /**
     * Waits for all threads on a list to finish.
     *
     * @param threads list of threads
     */
    public static void waitAll(List<Thread> threads) {
        for (Thread thread : threads) {
            try {
                thread.join();
            } catch (InterruptedException e) {
                e.printStackTrace(System.err);
            }
        }
    }

    /**
     * Clips the gradient to avoid gradient explosion.
     *
     * @param gradient the gradient
     * @return the clipped gradient
     */
    public static float clipGradient(double gradient) {
        return (float) Math.max(Math.min(gradient, GRADIENT_CLIP), -GRADIENT_CLIP);
    }

    public static String formatNumber(int params) {
        String[] prefixes = {"", "KB", "MB", "GB", "TB"};

        if (params == 0) {
            return "0";
        }

        int ciphers = (int) (Math.log10(params) / 3);

        double divisor = Math.pow(1000, ciphers);
        double normalized = params / divisor;

        return String.format("%.2f %s", normalized, prefixes[ciphers]);
    }
}