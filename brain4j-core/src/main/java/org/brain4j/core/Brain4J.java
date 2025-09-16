package org.brain4j.core;

import ch.qos.logback.classic.Level;
import org.brain4j.math.activation.Activation;
import org.brain4j.math.gpu.GpuContext;
import org.brain4j.math.gpu.device.Device;
import org.brain4j.math.gpu.device.DeviceUtils;
import org.brain4j.math.activation.Activations;
import org.brain4j.math.tensor.impl.GpuTensor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

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
 * @since 3.0
 * @author xEcho1337
 * @author Adversing
 */
public class Brain4J {

    private static boolean logging = true;

    public static String version() {
        return "3.0";
    }

    public static boolean logging() {
        return logging;
    }

    public static void disableLogging() {
        ch.qos.logback.classic.Logger root = (ch.qos.logback.classic.Logger) LoggerFactory.getLogger(Logger.ROOT_LOGGER_NAME);
        root.setLevel(Level.OFF);
        logging = false;
    }

    public static void enableLogging() {
        ch.qos.logback.classic.Logger root = (ch.qos.logback.classic.Logger) LoggerFactory.getLogger(Logger.ROOT_LOGGER_NAME);
        root.setLevel(Level.DEBUG);
        logging = true;
    }

    public static String availableDevices() {
        return String.join(", ", DeviceUtils.allDeviceNames());
    }

    public static void initKernels(Device device) {
        GpuTensor.initKernels(device);
    }

    public static Device firstDevice() {
        List<String> devices = DeviceUtils.allDeviceNames();

        if (devices.isEmpty()) {
            throw new IllegalStateException("No GPU-acceleration device has been found!");
        }

        return DeviceUtils.findDevice(devices.getFirst());
    }
    
    public static List<Device> allDevices() {
        List<Device> devices = new ArrayList<>();
        
        for (String device : DeviceUtils.allDeviceNames()) {
            devices.add(DeviceUtils.findDevice(device));
        }
        
        return devices;
    }

    public static Device findDevice(String deviceName) {
        return DeviceUtils.findDevice(deviceName);
    }
}
