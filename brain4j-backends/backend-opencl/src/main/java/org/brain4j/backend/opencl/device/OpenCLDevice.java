package org.brain4j.backend.opencl.device;

import org.brain4j.backend.device.CommandQueue;
import org.brain4j.backend.device.Device;
import org.jocl.CL;
import org.jocl.cl_command_queue;
import org.jocl.cl_context;
import org.jocl.cl_device_id;

import static org.jocl.CL.*;

public class OpenCLDevice implements Device {

    private final cl_device_id device;
    private final cl_context context;

    public OpenCLDevice(cl_device_id device, cl_context context) {
        this.device = device;
        this.context = context;
    }

    @Override
    public String name() {
        long[] size = new long[1];
        clGetDeviceInfo(device, CL_DEVICE_NAME, 0, null, size);

        byte[] buffer = new byte[(int) size[0]];
        clGetDeviceInfo(device, CL_DEVICE_NAME, buffer.length, org.jocl.Pointer.to(buffer), null);

        return new String(buffer).trim();
    }

    @Override
    public CommandQueue newCommandQueue() {
        cl_command_queue queue = clCreateCommandQueue(
            context, device, 0, null
        );
        return new OpenCLCommandQueue(queue);
    }

    public cl_device_id getDevice() {
        return device;
    }

    public cl_context getContext() {
        return context;
    }
}
