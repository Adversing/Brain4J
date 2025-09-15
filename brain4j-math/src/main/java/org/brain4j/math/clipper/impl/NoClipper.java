package org.brain4j.math.clipper.impl;

import org.brain4j.math.tensor.impl.CpuTensor;
import org.brain4j.math.tensor.impl.GpuTensor;
import org.brain4j.math.clipper.GradientClipper;

public class NoClipper implements GradientClipper {
    
    @Override
    public void clipCpu(CpuTensor grad) {

    }

    @Override
    public void clipGpu(GpuTensor grad) {

    }

    @Override
    public String kernelName() {
        return "";
    }
}