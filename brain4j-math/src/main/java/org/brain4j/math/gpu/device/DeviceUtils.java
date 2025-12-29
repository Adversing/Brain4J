package org.brain4j.math.gpu.device;

import org.brain4j.math.tensor.impl.GpuTensor;
import org.lwjgl.PointerBuffer;
import org.lwjgl.opencl.CL10;
import org.lwjgl.system.MemoryStack;

import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.nio.ByteBuffer;
import java.nio.IntBuffer;
import java.util.ArrayList;
import java.util.List;

import static java.nio.charset.StandardCharsets.UTF_8;

public class DeviceUtils {
    
    public static Device findDevice(String name) {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            IntBuffer buffer = stack.mallocInt(1);
            CL10.clGetPlatformIDs(null, buffer);

            int platformCount = buffer.get(0);

            if (platformCount == 0) {
                throw new RuntimeException("No OpenCL platforms found.");
            }

            PointerBuffer platforms = stack.mallocPointer(platformCount);
            CL10.clGetPlatformIDs(platforms, (IntBuffer) null);

            for (int i = 0; i < platforms.capacity(); i++) {
                long platform = platforms.get(i);
                int result = CL10.clGetDeviceIDs(platform, CL10.CL_DEVICE_TYPE_GPU, null, buffer);

                if (result != CL10.CL_SUCCESS) return null;

                int devicesCount = buffer.get(0);

                PointerBuffer devices = stack.mallocPointer(devicesCount);
                CL10.clGetDeviceIDs(platform, CL10.CL_DEVICE_TYPE_GPU, devices, (IntBuffer)null);

                if (name == null) {
                    return new Device(platform, devices.get(0));
                }

                for (int d = 0; d < devices.capacity(); d++) {
                    long device = devices.get(d);

                    if (!deviceName(stack, device).contains(name)) continue;

                    return new Device(platform, device);
                }
            }
        }

        return null;
    }

    public static String deviceName(long device) {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            return deviceName(stack, device);
        }
    }

    public static String deviceName(MemoryStack stack, long device) {
        PointerBuffer sizeBuf = stack.mallocPointer(1);
        CL10.clGetDeviceInfo(device, CL10.CL_DEVICE_NAME, (ByteBuffer) null, sizeBuf);

        long size = sizeBuf.get(0);

        ByteBuffer nameBuffer = stack.malloc((int) size);
        CL10.clGetDeviceInfo(device, CL10.CL_DEVICE_NAME, nameBuffer, null);

        return UTF_8.decode(nameBuffer).toString().trim();
    }

    public static List<String> allDeviceNames() {
        List<String> deviceNames = new ArrayList<>();

        try (MemoryStack stack = MemoryStack.stackPush()) {
            IntBuffer platformCounter = stack.mallocInt(1);
            CL10.clGetPlatformIDs(null, platformCounter);

            int numPlatforms = platformCounter.get(0);
            if (numPlatforms == 0) return deviceNames;

            PointerBuffer platforms = stack.mallocPointer(numPlatforms);
            CL10.clGetPlatformIDs(platforms, (IntBuffer) null);

            for (int i = 0; i < platforms.capacity(); i++) {
                long platform = platforms.get(i);

                IntBuffer di = stack.mallocInt(1);
                CL10.clGetDeviceIDs(platform, CL10.CL_DEVICE_TYPE_ALL, null, di);

                int numDevices = di.get(0);
                if (numDevices == 0) continue;

                PointerBuffer devices = stack.mallocPointer(numDevices);
                CL10.clGetDeviceIDs(platform, CL10.CL_DEVICE_TYPE_ALL, devices, (IntBuffer) null);

                for (int d = 0; d < devices.capacity(); d++) {
                    long dev = devices.get(d);
                    deviceNames.add(deviceName(stack, dev));
                }
            }
        }

        return deviceNames;
    }

    public static String readKernelSource(String resourcePath) {
        try (InputStream input = GpuTensor.class.getResourceAsStream(resourcePath)) {
            if (input == null) {
                throw new IllegalArgumentException("Resource not found: " + resourcePath);
            }
            return new String(input.readAllBytes(), UTF_8);
        } catch (IOException e) {
            throw new UncheckedIOException("Failed to read kernel source from: " + resourcePath, e);
        }
    }

    public static long createBuildProgram(Device device, String path) {
        String source = readKernelSource(path);
        long  context = device.context();

        long program = CL10.clCreateProgramWithSource(context, source, null);
        
        if (program == 0L) {
            throw new RuntimeException("clCreateProgramWithSource returned NULL");
        }

        int buildErr = CL10.clBuildProgram(program, device.device(), "", null, 0);
        if (buildErr != CL10.CL_SUCCESS) {
            String buildLog = getBuildLog(program, device.device());
            throw new RuntimeException("clBuildProgram failed: " + buildErr + " (" + getErrorCode(buildErr) + ")\nBuild log:\n" + buildLog);
        }

        return program;
    }

    private static String getBuildLog(long program, long device) {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            PointerBuffer sizeBuf = stack.mallocPointer(1);
            CL10.clGetProgramBuildInfo(program, device, CL10.CL_PROGRAM_BUILD_LOG, (ByteBuffer) null, sizeBuf);

            long size = sizeBuf.get(0);
            if (size <= 1) {
                return "(no build log available)";
            }

            ByteBuffer logBuffer = stack.malloc((int) size);
            CL10.clGetProgramBuildInfo(program, device, CL10.CL_PROGRAM_BUILD_LOG, logBuffer, null);

            return UTF_8.decode(logBuffer).toString().trim();
        } catch (Exception e) {
            return "(failed to get build log: " + e.getMessage() + ")";
        }
    }

    public static boolean isSimdAvailable() {
        return ModuleLayer.boot().findModule("jdk.incubator.vector").isPresent();
    }

    public static String getErrorCode(int code) {
        return switch (code) {
            case 0 -> "CL_SUCCESS";
            case -1 -> "CL_DEVICE_NOT_FOUND";
            case -2 -> "CL_DEVICE_NOT_AVAILABLE";
            case -3 -> "CL_COMPILER_NOT_AVAILABLE";
            case -4 -> "CL_MEM_OBJECT_ALLOCATION_FAILURE";
            case -5 -> "CL_OUT_OF_RESOURCES";
            case -6 -> "CL_OUT_OF_HOST_MEMORY";
            case -7 -> "CL_PROFILING_INFO_NOT_AVAILABLE";
            case -8 -> "CL_MEM_COPY_OVERLAP";
            case -9 -> "CL_IMAGE_FORMAT_MISMATCH";
            case -10 -> "CL_IMAGE_FORMAT_NOT_SUPPORTED";
            case -11 -> "CL_BUILD_PROGRAM_FAILURE";
            case -12 -> "CL_MAP_FAILURE";
            case -13 -> "CL_MISALIGNED_SUB_BUFFER_OFFSET";
            case -14 -> "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";

            // compile/link extensions
            case -15 -> "CL_COMPILE_PROGRAM_FAILURE";
            case -16 -> "CL_LINKER_NOT_AVAILABLE";
            case -17 -> "CL_LINK_PROGRAM_FAILURE";
            case -18 -> "CL_DEVICE_PARTITION_FAILED";
            case -19 -> "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

            // invalid values
            case -30 -> "CL_INVALID_VALUE";
            case -31 -> "CL_INVALID_DEVICE_TYPE";
            case -32 -> "CL_INVALID_PLATFORM";
            case -33 -> "CL_INVALID_DEVICE";
            case -34 -> "CL_INVALID_CONTEXT";
            case -35 -> "CL_INVALID_QUEUE_PROPERTIES";
            case -36 -> "CL_INVALID_COMMAND_QUEUE";
            case -37 -> "CL_INVALID_HOST_PTR";
            case -38 -> "CL_INVALID_MEM_OBJECT";
            case -39 -> "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
            case -40 -> "CL_INVALID_IMAGE_SIZE";
            case -41 -> "CL_INVALID_SAMPLER";
            case -42 -> "CL_INVALID_BINARY";
            case -43 -> "CL_INVALID_BUILD_OPTIONS";
            case -44 -> "CL_INVALID_PROGRAM";
            case -45 -> "CL_INVALID_PROGRAM_EXECUTABLE";
            case -46 -> "CL_INVALID_KERNEL_NAME";
            case -47 -> "CL_INVALID_KERNEL_DEFINITION";
            case -48 -> "CL_INVALID_KERNEL";
            case -49 -> "CL_INVALID_ARG_INDEX";
            case -50 -> "CL_INVALID_ARG_VALUE";
            case -51 -> "CL_INVALID_ARG_SIZE";
            case -52 -> "CL_INVALID_KERNEL_ARGS";
            case -53 -> "CL_INVALID_WORK_DIMENSION";
            case -54 -> "CL_INVALID_WORK_GROUP_SIZE";
            case -55 -> "CL_INVALID_WORK_ITEM_SIZE";
            case -56 -> "CL_INVALID_GLOBAL_OFFSET";
            case -57 -> "CL_INVALID_EVENT_WAIT_LIST";
            case -58 -> "CL_INVALID_EVENT";
            case -59 -> "CL_INVALID_OPERATION";
            case -60 -> "CL_INVALID_GL_OBJECT";
            case -61 -> "CL_INVALID_BUFFER_SIZE";
            case -62 -> "CL_INVALID_MIP_LEVEL";
            case -63 -> "CL_INVALID_GLOBAL_WORK_SIZE";
            case -64 -> "CL_INVALID_PROPERTY";
            case -65 -> "CL_INVALID_IMAGE_DESCRIPTOR";
            case -66 -> "CL_INVALID_COMPILER_OPTIONS";
            case -67 -> "CL_INVALID_LINKER_OPTIONS";
            case -68 -> "CL_INVALID_DEVICE_PARTITION_COUNT";

            default -> "Unknown error code (" + code + ")";
        };
    }

    public static void checkError(String profiler, int err) {
        if (err == 0) return;

        String errorCode = getErrorCode(err);
        String error = String.format("OpenCL(%s) - %s", profiler, errorCode);

        throw new RuntimeException(error);
    }
}
