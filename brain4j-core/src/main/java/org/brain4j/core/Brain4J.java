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
        logging = false;
    }

    public static void enableLogging() {
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
