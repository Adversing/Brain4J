package org.brain4j.core.clipper.impl;

import org.brain4j.common.tensor.impl.CpuTensor;
import org.brain4j.common.tensor.impl.GpuTensor;
import org.brain4j.core.clipper.GradientClipper;
import org.brain4j.core.importing.proto.ProtoModel;

public class NoClipper implements GradientClipper {
    
    @Override
    public void serialize(ProtoModel.Clipper.Builder builder) {
    
    }
    
    @Override
    public void deserialize(ProtoModel.Clipper protoClipper) {
    
    }
    
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