package org.brain4j.core.transformer.attention;

import org.brain4j.math.Commons;
import org.brain4j.math.Tensors;
import org.brain4j.math.gpu.device.Device;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.weightsinit.WeightInitialization;
import org.brain4j.core.clipper.GradientClipper;
import org.brain4j.core.clipper.impl.NoClipper;
import org.brain4j.core.importing.proto.ProtoModel;
import org.brain4j.core.importing.proto.SerializeUtils;
import org.brain4j.core.training.StatesCache;
import org.brain4j.core.training.optimizer.Optimizer;
import org.brain4j.core.training.updater.Updater;
import org.brain4j.core.transformer.attention.head.AttentionHead;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class MultiHeadAttention {

    protected GradientClipper clipper;
    protected Tensor outProjWeights;
    protected List<AttentionHead> heads;
    protected int headCount;
    protected int embeddingDim;
    protected int headDimension;
    
    public MultiHeadAttention() {
        this.heads = new ArrayList<>();
    }
    
    public MultiHeadAttention(GradientClipper clipper, int headCount, int embeddingDim) {
        this.clipper = clipper;
        this.headCount = headCount;
        this.embeddingDim = embeddingDim;

        if (embeddingDim % headCount != 0) {
            throw new IllegalArgumentException("Embedding dimension must be divisible by head count! (%s %% %s = %s)"
                    .formatted(embeddingDim, headCount, embeddingDim % headCount));
        }

        this.headDimension = embeddingDim / headCount;
        this.heads = new ArrayList<>();
        this.outProjWeights = Tensors.matrix(embeddingDim, embeddingDim).withGrad();

        initializeHeads();
    }
    
    public AttentionHead createAttentionHead() {
        return new AttentionHead(clipper, embeddingDim, headDimension);
    }

    public void toDevice(Device device) {
        for (AttentionHead head : heads) {
            head.toDevice(device);
        }
    }

    public void compile(Random generator, WeightInitialization weightInit) {
        for (AttentionHead head : heads) {
            head.initWeights(generator, weightInit);
        }

        this.outProjWeights.map(x -> weightInit.generate(generator, embeddingDim, embeddingDim));
    }
    
    protected void initializeHeads() {
        for (int i = 0; i < headCount; i++) {
            heads.add(createAttentionHead());
        }
    }

    public Tensor attend(StatesCache cache, Tensor input) {
        Tensor[] outputs = new Tensor[heads.size()];

        for (int i = 0; i < heads.size(); i++) {
            outputs[i] = heads.get(i).attend(cache, input);
        }

        Tensor result = Tensors.concatGrad(List.of(outputs));
        return result.matmulGrad(outProjWeights);
    }

    public int totalWeights() {
        return heads.stream().mapToInt(AttentionHead::totalWeights).sum();
    }

    public List<AttentionHead> heads() {
        return heads;
    }

    public void backward(Updater updater, Optimizer optimizer) {
        for (AttentionHead head : heads) {
            head.backward(updater, optimizer);
        }
    }

    public ProtoModel.MultiHeadAttention serialize() {
        ProtoModel.MultiHeadAttention.Builder builder =
            ProtoModel.MultiHeadAttention.newBuilder()
                .setOutWeight(SerializeUtils.serializeTensor("out_weight", outProjWeights))
                .putAttrs("gradient_clipper", SerializeUtils.value(clipper.getClass().getName()))
                .putAttrs("head_count", SerializeUtils.value(headCount))
                .putAttrs("embedding_dim", SerializeUtils.value(embeddingDim))
                .putAttrs("head_dimension", SerializeUtils.value(headDimension));
        
        for (AttentionHead head : heads()) {
            builder.addHeads(head.serialize());
        }
        
        return builder.build();
    }
    
    public void deserialize(ProtoModel.MultiHeadAttention attention) {
        this.outProjWeights = SerializeUtils.deserializeTensor(attention.getOutWeight());
        this.clipper = Commons.newInstance(SerializeUtils.attribute(attention.getAttrsMap(), "gradient_clipper", NoClipper.class.getName()));
        this.headCount = SerializeUtils.attribute(attention.getAttrsMap(), "head_count", 0);
        this.embeddingDim = SerializeUtils.attribute(attention.getAttrsMap(), "embedding_dim", 0);
        this.headDimension = SerializeUtils.attribute(attention.getAttrsMap(), "head_dimension", 0);
        this.heads = new ArrayList<>();
        
        for (ProtoModel.AttentionHead protoHead : attention.getHeadsList()) {
            AttentionHead head = createAttentionHead();
            head.deserialize(protoHead);
            heads.add(head);
        }
    }

    public void resetGrad() {
        for (AttentionHead head : heads()) {
            head.resetGrad();
        }

        outProjWeights.zerograd();
    }

    public GradientClipper clipper() {
        return clipper;
    }
    
    public void setClipper(GradientClipper clipper) {
        this.clipper = clipper;
    }
    
    public Tensor outProjWeights() {
        return outProjWeights;
    }
    
    public void setOutProjWeights(Tensor outProjWeights) {
        this.outProjWeights = outProjWeights;
    }
    
    public int headCount() {
        return headCount;
    }
    
    public void setHeadCount(int headCount) {
        this.headCount = headCount;
    }
    
    public int embeddingDim() {
        return embeddingDim;
    }
    
    public void setEmbeddingDim(int embeddingDim) {
        this.embeddingDim = embeddingDim;
    }
    
    public int headDimension() {
        return headDimension;
    }
    
    public void setHeadDimension(int headDimension) {
        this.headDimension = headDimension;
    }
}
