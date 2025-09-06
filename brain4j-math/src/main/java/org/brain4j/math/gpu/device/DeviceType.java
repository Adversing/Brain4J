package org.brain4j.math.gpu.device;

public enum DeviceType {

    CPU(1 << 1),
    GPU(1 << 2);

    private final int mask;

    DeviceType(int mask) {
        this.mask = mask;
    }

    public int getMask() {
        return mask;
    }
}
