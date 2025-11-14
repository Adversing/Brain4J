package org.brain4j.math.gpu.device;

import org.brain4j.math.tensor.impl.GpuTensor;
import org.jocl.*;
import org.lwjgl.PointerBuffer;
import org.lwjgl.opencl.CL10;
import org.lwjgl.system.MemoryStack;

import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.nio.IntBuffer;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

import static org.jocl.CL.*;

public class DeviceUtils {

    public static Device findDevice(String name) {
//        int gpuMask = 1 << 2;
//        int[] numPlatformsArray = new int[1];
//
//        try (MemoryStack stack = MemoryStack.stackPush()) {
//            IntBuffer buffer = stack.mallocInt(1);
//            CL10.clGetPlatformIDs(null, buffer);
//
//            int platformCount = buffer.get(0);
//
//            if (platformCount == 0) {
//                throw new RuntimeException("No OpenCL platforms found.");
//            }
//
//            PointerBuffer platforms = stack.mallocPointer(platformCount);
//            CL10.clGetPlatformIDs(platforms, (IntBuffer) null);
//
//            for (int i = 0; i < platforms.capacity(); i++) {
//                long platform = platforms.get(i);
//                int result = CL10.clGetDeviceIDs(platform, CL10.CL_DEVICE_TYPE_GPU, null, buffer);
//
//                if (result != CL10.CL_SUCCESS) return null;
//
//                int devicesCount = buffer.get(0);
//
//                PointerBuffer devices = stack.mallocPointer(devicesCount);
//                CL10.clGetDeviceIDs(platform, CL10.CL_DEVICE_TYPE_GPU, devices, (IntBuffer)null);
//
//                if (name == null) {
//                    return new Device(platform, devices[0]);
//                }
//
//                for (int d = 0; d < devices.capacity(); d++) {
//                    long device = devices.get(d);
//
//                    if (!deviceName(device).contains(name)) continue;
//
//                    return new Device(platform, dev);
//                }
//                for (long dev : devices) {
//                    if (deviceName(dev).contains(name)) {
//                        return new Device(platform, dev);
//                    }
//                }
//            }
//        }

        return null;
    }

    public static String deviceName(cl_device_id device) {
        long[] size = new long[1];
        // CL10.clGetDeviceInfo(device, CL10.CL_DEVICE_NAME, null, null);
        clGetDeviceInfo(device, CL10.CL_DEVICE_NAME, 0, null, size);

        byte[] buffer = new byte[(int) size[0]];
        clGetDeviceInfo(device, CL_DEVICE_NAME, buffer.length, Pointer.to(buffer), null);

        return new String(buffer, 0, buffer.length - 1).trim();
    }

    public static List<String> allDeviceNames() {
        List<String> deviceNames = new ArrayList<>();

        int[] numPlatformsArray = new int[1];
        clGetPlatformIDs(0, null, numPlatformsArray);
        int numPlatforms = numPlatformsArray[0];

        cl_platform_id[] platforms = new cl_platform_id[numPlatforms];
        clGetPlatformIDs(platforms.length, platforms, null);

        for (cl_platform_id platform : platforms) {
            int[] numDevicesArray = new int[1];
            int result = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, null, numDevicesArray);

            if (result != CL_SUCCESS) continue;

            int numDevices = numDevicesArray[0];

            cl_device_id[] devices = new cl_device_id[numDevices];
            clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, numDevices, devices, null);

            for (cl_device_id dev : devices) {
                deviceNames.add(deviceName(dev));
            }
        }

        return deviceNames;
    }

    public static String readKernelSource(String resourcePath) {
        try (InputStream input = GpuTensor.class.getResourceAsStream(resourcePath)) {
            if (input == null) {
                throw new IllegalArgumentException("Resource not found: " + resourcePath);
            }
            return new String(input.readAllBytes(), StandardCharsets.UTF_8);
        } catch (IOException e) {
            throw new UncheckedIOException("Failed to read kernel source from: " + resourcePath, e);
        }
    }

    public static cl_program createBuildProgram(cl_context context, String path) {
        String source = readKernelSource(path);

        cl_program program = clCreateProgramWithSource(context, 1, new String[]{source}, null, null);
        clBuildProgram(program, 0, null, null, null, null);

        return program;
    }

    public static boolean isSimdAvailable() {
        return ModuleLayer.boot().findModule("jdk.incubator.vector").isPresent();
    }
}
