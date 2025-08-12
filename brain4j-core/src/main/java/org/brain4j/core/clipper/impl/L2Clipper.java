package org.brain4j.core.clipper.impl;

import org.brain4j.common.tensor.Tensor;
import org.brain4j.common.tensor.impl.CpuTensor;
import org.brain4j.common.tensor.impl.GpuTensor;
import org.brain4j.core.clipper.GradientClipper;
import org.brain4j.core.importing.proto.ProtoModel;
import org.brain4j.core.importing.proto.SerializeUtils;

public class L2Clipper implements GradientClipper {

    private double scale;

    public L2Clipper(double scale) { this.scale = scale; }
    
    @Override
    public void serialize(ProtoModel.Clipper.Builder builder) {
        builder.putAttrs("scale", SerializeUtils.value(scale));
    }
    
    @Override
    public void deserialize(ProtoModel.Clipper protoClipper) {
        this.scale = SerializeUtils.attribute(protoClipper.getAttrsMap(), "scale", 0);
    }
    
    @Override
    public void clipCpu(CpuTensor grad) {
        double threshold = scale * Math.sqrt(grad.elements());
        double norm = sumOfSquares(grad);

        if (norm > threshold) {
            float scaleFactor = (float) (threshold / norm);
            grad.mul(scaleFactor);
        }
    }

    @Override
    public void clipGpu(GpuTensor grad) {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    @Override
    public String kernelName() {
        return "l2_clip";
    }

    public double sumOfSquares(Tensor input) {
        double sumOfSquares = 0.0;

        for (int i = 0; i < input.elements(); i++) {
            sumOfSquares += Math.pow(input.data()[i], 2);
        }

        return Math.sqrt(sumOfSquares);
    }
}