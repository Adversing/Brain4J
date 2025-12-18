package org.brain4j.math.commons;

import org.brain4j.math.tensor.Tensor;

import java.lang.reflect.Constructor;
import java.time.Duration;

/**
 * General utility methods used across the math module.
 *
 * <p>This class provides various helper functions including:
 * <ul>
 *   <li>Progress bar and formatting utilities for training output
 *   <li>Numeric utilities (clamp, modulo, float16 conversion)
 *   <li>Reflection helpers for instantiating classes
 *   <li>Tensor classification helpers
 * </ul>
 *
 * <p>Many methods in this class are used internally by the framework's
 * training and evaluation components.
 *
 * @author xEcho1337
 */
public class Commons {

    public static String HEADER_CHAR = "━"; 
    private static final int[] EXP_TABLE = new int[64];
    private static final int[] MANT_TABLE = new int[2048];
    private static final int[] OFF_TABLE = new int[64];

    static {
        precomputeTables();
    }

    private static void precomputeTables() {
        for (int i = 0; i < 64; i++) {
            int e = i - 15;
            if (i == 0) {
                EXP_TABLE[i] = 0;
                OFF_TABLE[i] = 0;
            } else if (i == 31) {
                EXP_TABLE[i] = 0xFF << 23;
                OFF_TABLE[i] = 0;
            } else {
                EXP_TABLE[i] = (e + 127) << 23;
                OFF_TABLE[i] = 1024;
            }
        }

        MANT_TABLE[0] = 0;
        for (int i = 1; i < 2048; i++) {
            int m = i & 0x3FF;
            int e = i >> 10;
            if (e == 0) {
                int mantissa = m;
                int exp = -1;
                do {
                    mantissa <<= 1;
                    exp--;
                } while ((mantissa & 0x400) == 0);
                mantissa &= 0x3FF;
                MANT_TABLE[i] = (mantissa << 13) | ((exp + 1 + 127) << 23);
            } else {
                MANT_TABLE[i] = m << 13;
            }
        }
    }

    public static String createProgressBar(
        double percent,
        int characterCount,
        String barCharacter,
        String emptyCharacter
    ) {
        if (percent < 0 || percent > 1) {
            throw new IllegalArgumentException("Percent must be between 0 and 1!");
        }

        int fill = (int) Math.round(percent * characterCount);
        int remaining = characterCount - fill;

        return barCharacter.repeat(fill) + emptyCharacter.repeat(remaining);
    }

    public static float f16ToFloat(short half) {
        int bits = half & 0xFFFF;
        int sign = bits >>> 15;
        int exp = (bits >>> 10) & 0x1F;
        int mantissa = bits & 0x3FF;
        int floatBits = (sign << 31) | EXP_TABLE[exp] | MANT_TABLE[mantissa + OFF_TABLE[exp]];
        return Float.intBitsToFloat(floatBits);
    }

    public static int nextPowerOf2(int n) {
        if (n <= 0) return 1;
        n--;
        n |= n >>> 1;
        n |= n >>> 2;
        n |= n >>> 4;
        n |= n >>> 8;
        n |= n >>> 16;
        return n + 1;
    }

    public static boolean isPowerOf2(int n) {
        return n > 0 && (n & (n - 1)) == 0;
    }

    public static String formatDuration(double seconds) {
        double millis = seconds * 1000;
        Duration duration = Duration.ofMillis((long) millis);

        if (seconds < 1) {
            return String.format("%.2fms", millis);
        }

        if (seconds < 60) {
            return String.format("%.2fs", seconds);
        }

        long minutes = duration.toMinutesPart();
        long secs = duration.toSecondsPart();

        return (secs == 0)
                ? String.format("%dm", minutes)
                : String.format("%dm%ds", minutes, secs);
    }

    /**
     * Parses a classification output tensor into an enum constant.
     * <p>
     * Takes a 1D tensor of class probabilities/logits and returns the enum constant
     * corresponding to the highest score (argmax). Useful for classification tasks
     * where the output classes are represented by an enum.
     *
     * @param outputs the model output tensor (must be 1D)
     * @param clazz the enum class containing the possible classes
     * @param <T> the enum type
     * @return the predicted enum class
     * @throws IllegalArgumentException if outputs is not 1-dimensional
     */
    public static <T extends Enum<T>> T parse(Tensor outputs, Class<T> clazz) {
        if (outputs.rank() != 1) {
            throw new IllegalArgumentException("Output tensor must be 1-dimensional!");
        }

        return clazz.getEnumConstants()[outputs.argmax()];
    }

    /**
     * Creates a header string with centered text.
     * <p>
     * The text is centered between repeating characters, useful for
     * creating section headers in console output.
     *
     * @param middleText the text to center
     * @param character the character to repeat around the text
     * @return a formatted header string
     */
    public static String getHeader(String middleText, String character) {
        int maxLength = 70;
        int middleLength = middleText.length();
        int repeatedLength = (maxLength - middleLength) / 2;

        String repeated = character.repeat(repeatedLength);
        String result = repeated + middleText + repeated;

        if (result.length() < maxLength) {
            result += character;
        }

        return result + "\n";
    }

    public static String formatNumber(long params) {
        String[] prefixes = {"B", "KB", "MB", "GB", "TB"};

        if (params == 0) return "0";

        int exponent = (int) (Math.log10(params) / 3);

        double divisor = Math.pow(1000, exponent);
        double normalized = params / divisor;

        if (exponent > prefixes.length) {
            throw new UnsupportedOperationException("Input number too big!");
        }

        return "%.2f %s".formatted(normalized, prefixes[exponent]);
    }

    @SuppressWarnings("all")
    public static <T> T newInstance(String classPath) {
        // Support for versions prior to 2.9
        classPath = classPath.replace("net.echo.brain4j", "org.brain4j.core");
        classPath = classPath.replace("net.echo.math", "org.brain4j.math");
        
        try {
            Class<?> clazz = Class.forName(classPath);
            return (T) newInstance(clazz);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
    
    @SuppressWarnings("all")
    public static <T> T newInstance(Class<T> clazz) {
        try {
            Constructor<?> constructor = clazz.getDeclaredConstructor();
            constructor.setAccessible(true);
            
            return (T) constructor.newInstance();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
    
    public static double clamp(float value, double minimum, double maximum) {
        return Math.min(Math.max(value, minimum), maximum);
    }
    
    public static double clamp(double value, double minimum, double maximum) {
        return Math.min(Math.max(value, minimum), maximum);
    }

    public static float[] int2float(int[] array) {
        float[] result = new float[array.length];

        for (int i = 0; i < array.length; i++) {
            result[i] = array[i];
        }

        return result;
    }
    
    public static int mod(int x, int m) {
        int r = x % m;
        return r < 0 ? r + m : r;
    }
}