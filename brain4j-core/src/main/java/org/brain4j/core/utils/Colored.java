package org.brain4j.core.utils;

import org.brain4j.core.Brain4J;

import java.util.Map;

import static org.brain4j.math.Constants.*;
import static org.brain4j.math.Constants.GRAY;
import static org.brain4j.math.Constants.LIGHT_BLUE;
import static org.brain4j.math.Constants.LIGHT_GREEN;
import static org.brain4j.math.Constants.MAGENTA;

public class Colored {
    
    private static final Map<String, String> COLORS = Map.of(
        "yellow", LIGHT_YELLOW,
        "white", WHITE,
        "blue", LIGHT_BLUE,
        "green", LIGHT_GREEN,
        "gray", GRAY,
        "magenta", MAGENTA,
        "reset", RESET
    );
    
    /**
     * Renders a formatted string with inline color tags.
     * <p>
     * The method first applies {@link String#format(String, Object...)} using the
     * provided arguments, then replaces color placeholders of the form
     * {@code <color>} with their corresponding ANSI escape codes.
     * <p>
     * A terminal reset code is always appended at the end of the returned string.
     *
     * @param template the format string containing optional color tags
     * @param args arguments referenced by the format specifiers
     * @return the rendered string with ANSI color codes applied
     */
    public static String renderText(String template, Object... args) {
        String formatted = String.format(template, args);
        
        for (var entry : COLORS.entrySet()) {
            String value = Brain4J.isDisableColors() ? "" : entry.getValue();
            formatted = formatted.replace("<" + entry.getKey() + ">", value);
        }
        
        return formatted;
    }
}
