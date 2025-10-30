package org.brain4j.core;

import org.brain4j.math.gpu.device.Device;
import org.brain4j.math.gpu.device.DeviceUtils;
import org.brain4j.math.tensor.impl.GpuTensor;

import java.util.ArrayList;
import java.util.List;

/**
 * Entry point for the Brain4J machine learning framework.
 * <p>
 * The {@code Brain4J} class provides central static methods to access core functionalities
 * such as framework initialization, device discovery, and version management.
 * <p>
 * This class is designed to serve as the main API access point, and will be extended
 * in the future to include global configuration, logging, and other utilities.
 *
 * <p><b>Example usage:</b>
 * <pre>{@code
 * System.out.println("Brain4J version: " + Brain4J.version());
 * Brain4J.enableLogging();
 * System.out.println("Available devices: " + Brain4J.availableDevices());
 * }</pre>
 *
 * @author xEcho1337
 * @author Adversing
 */
public class Brain4J {

    private static boolean logging = true;
    private static int precision = 4;

    /**
     * Returns the current version of the Brain4J framework.
     *
     * @return the version string (e.g. "3.0")
     */
    public static String version() {
        return "3.0";
    }

    /**
     * Indicates whether training progress and system information
     * are currently being logged to the console.
     *
     * @return {@code true} if logging is enabled; {@code false} otherwise
     */
    public static boolean logging() {
        return logging;
    }

    /**
     * Disables console logging for all training sessions.
     * <p>
     * This can improve performance slightly in cases where
     * logging output is not needed.
     */
    public static void disableLogging() {
        logging = false;
    }

    /**
     * Enables console logging for all training sessions.
     * <p>
     * When enabled, Brain4J will print progress information
     * such as epoch, batch index, and loss values.
     */
    public static void enableLogging() {
        logging = true;
    }

    /**
     * Returns the numeric precision used when displaying
     * loss or metric values during training.
     *
     * @return the number of decimal digits displayed (default: 4)
     */
    public static int precision() {
        return precision;
    }

    /**
     * Sets the numeric precision used when printing loss values.
     * <p>
     * This affects how many decimal digits are shown in logs
     * and formatted console outputs.
     *
     * @param precision the number of digits to display after the decimal point
     */
    public static void setPrecision(int precision) {
        Brain4J.precision = precision;
    }

    /**
     * Returns a comma-separated list of all available GPU devices
     * detected by the OpenCL backend.
     * <p>
     * If no devices are available, an empty string is returned.
     *
     * @return a comma-separated list of device names
     */
    public static String availableDevices() {
        return String.join(", ", DeviceUtils.allDeviceNames());
    }

    /**
     * Initializes GPU kernels on the specified device.
     * <p>
     * This method compiles and loads all GPU-side kernels used
     * by {@link GpuTensor} operations. It should be called before
     * performing any GPU computation if not done automatically.
     *
     * @param device the target GPU device to initialize
     */
    public static void initKernels(Device device) {
        GpuTensor.initKernels(device);
    }

    /**
     * Returns the first available GPU device detected on the system.
     * <p>
     * This method is useful when the system contains a single GPU
     * or when the default device is sufficient for computation.
     *
     * @return the first detected {@link Device}
     * @throws IllegalStateException if no GPU devices are found
     */
    public static Device firstDevice() {
        List<String> devices = DeviceUtils.allDeviceNames();

        if (devices.isEmpty()) {
            throw new IllegalStateException("No GPU-acceleration device has been found!");
        }

        return DeviceUtils.findDevice(devices.getFirst());
    }

    /**
     * Returns a list of all GPU devices available to the framework.
     * <p>
     * Each {@link Device} object represents a physical or logical
     * compute device accessible via OpenCL.
     *
     * @return a list of all available {@link Device} instances
     */
    public static List<Device> allDevices() {
        List<Device> devices = new ArrayList<>();

        for (String device : DeviceUtils.allDeviceNames()) {
            devices.add(DeviceUtils.findDevice(device));
        }

        return devices;
    }

    /**
     * Finds a specific GPU device by its name.
     * <p>
     * The search is case-sensitive and matches the full name
     * returned by {@link DeviceUtils#allDeviceNames()}.
     *
     * @param deviceName the name of the device to look for
     * @return the corresponding {@link Device} instance, or {@code null} if not found
     */
    public static Device findDevice(String deviceName) {
        return DeviceUtils.findDevice(deviceName);
    }
}