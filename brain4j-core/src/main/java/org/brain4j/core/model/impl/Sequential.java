package org.brain4j.core.model.impl;

import org.brain4j.core.layer.Layer;
import org.brain4j.core.layer.impl.utility.InputLayer;
import org.brain4j.core.loss.LossFunction;
import org.brain4j.core.loss.impl.BinaryCrossEntropy;
import org.brain4j.core.model.Model;
import org.brain4j.core.model.ModelSpecs;
import org.brain4j.core.training.wrappers.EvaluationResult;
import org.brain4j.math.Tensors;
import org.brain4j.math.commons.Batch;
import org.brain4j.math.commons.Commons;
import org.brain4j.math.data.ListDataSource;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.gpu.device.Device;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.index.Range;

import java.text.DecimalFormat;
import java.util.*;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;
import java.util.random.RandomGenerator;
import java.util.stream.IntStream;

public class Sequential implements Model {
    
    private final ModelSpecs specs;
    private final List<Layer> layers;
    private final Device device;
    private final long seed;
    
    public Sequential(ModelSpecs specs, Device device, long seed) {
        this.specs = specs;
        this.layers = specs.buildLayerList();
        this.device = device;
        this.seed = seed;
        
        if (!(layers.getFirst() instanceof InputLayer)) {
            throw new IllegalArgumentException("First layer in the model must be an InputLayer instance!");
        }
        
        initLayers();
    }
    
    @Override
    public Tensor[] predict(StatesCache cache, Tensor... inputs) {
        Tensor[] buffer = new Tensor[inputs.length];
        
        for (int i = 0; i < buffer.length; i++) {
            Tensor input = inputs[i];
            
            if (input == null || input.rank() == 0) {
                throw new IllegalArgumentException("Input at " + i + " is either null or has dimension of 0!");
            }
            
            if (input.rank() < 2) {
                input = input.reshape(1, input.elements()); // reshape to [batch, input_size]
            }
            
            buffer[i] = cache.training() ? input.withGrad() : input;
        }
        
        for (Layer layer : layers) {
            buffer = layer.forward(cache, buffer);
        }
        
        return buffer;
    }
    
    @Override
    public EvaluationResult evaluate(ListDataSource dataSource, LossFunction lossFunction) {
        int classes = Math.max(2, dataSource.getSamples().getFirst().label().elements());
        Map<Integer, Tensor> classifications = new HashMap<>();
        
        for (int i = 0; i < classes; i++) {
            classifications.put(i, Tensors.zeros(classes));
        }
        
        AtomicReference<Double> totalLoss = new AtomicReference<>(0.0);
        
        dataSource.reset();
        
        while (dataSource.hasNext()) {
            Batch batch = dataSource.nextBatch();
            makeEvaluation(batch, classifications, totalLoss, lossFunction);
        }
        
        return new EvaluationResult(totalLoss.get() / dataSource.getSize(), classes, classifications);
    }
    
    @Override
    public Model fork(Device device) {
        return new Sequential(specs, device, seed);
    }
    
    @Override
    public Device getDevice() {
        return device;
    }
    
    @Override
    public void summary() {
        StringBuilder stats = new StringBuilder();
        DecimalFormat format = new DecimalFormat("#,###");
        
        String pattern = "%-7s %-20s %-12s %-15s %-15s\n";
        String divider = Commons.getHeader(" Architecture ", Commons.HEADER_CHAR);
        
        stats.append(divider);
        stats.append(pattern.formatted("Index", "Layer Type", "Parameters", "Shape", "Activation")).append("\n");
        
        AtomicLong totalWeights = new AtomicLong(0);
        AtomicLong totalBiases = new AtomicLong(0);
        
        append(pattern, stats, format, totalWeights, totalBiases);
        
        long weightsCount = totalWeights.get();
        long biasesCount = totalBiases.get();
        
        long params = weightsCount + biasesCount;
        
        String parameters = format.format(params);
        String weights = format.format(totalWeights);
        String biases = format.format(totalBiases);
        
        byte floatSize = Float.BYTES; // 4 bytes
        String sizeOfParams = Commons.formatNumber(params * floatSize);
        String sizeOfWeights = Commons.formatNumber(weightsCount * floatSize);
        String sizeOfBiases = Commons.formatNumber(biasesCount * floatSize);
        
        stats.append(Commons.getHeader(" Recap ", Commons.HEADER_CHAR));
        stats.append("Total weights: %s (%s)\n".formatted(weights, sizeOfWeights));
        stats.append("Total biases: %s (%s)\n".formatted(biases, sizeOfBiases));
        stats.append("Total parameters: %s (%s)\n".formatted(parameters, sizeOfParams));
        stats.append(Commons.getHeader("", Commons.HEADER_CHAR));
        
        Arrays.stream(stats.toString().split("\n")).forEach(System.out::println);
    }
    
    @Override
    public ModelSpecs getSpecs() {
        return specs;
    }
    
    public List<Layer> getLayers() {
        return Collections.unmodifiableList(layers);
    }
    
    private void makeEvaluation(
        Batch batch,
        Map<Integer, Tensor> classifications,
        AtomicReference<Double> totalLoss,
        LossFunction lossFunction
    ) {
        Tensor[] inputs = batch.getFirst();
        Tensor[] labels = batch.getSecond();
        
        Tensor[] outputs = predict(new StatesCache(false, device), inputs);
        
        for (Tensor input : inputs) {
            int batchSize = input.shape(0);
            
            for (int i = 0; i < outputs.length; i++) {
                Tensor output = outputs[i].cpu();
                Tensor label = labels[i].cpu();
                
                for (int b = 0; b < batchSize; b++) {
                    Range range = Range.point(b);
                    
                    Tensor sampleOutput = output.slice(range).flatten();
                    Tensor sampleLabel = label.slice(range).flatten();
                    
                    int predIndex = sampleOutput.argmax();
                    int targetIndex = sampleLabel.argmax();
                    
                    if (sampleOutput.elements() == 1 && lossFunction instanceof BinaryCrossEntropy) {
                        predIndex = sampleOutput.get(0) > 0.5 ? 1 : 0;
                        targetIndex = (int) sampleLabel.get(0);
                    }
                    
                    double loss = lossFunction.calculate(sampleLabel, sampleOutput);
                    totalLoss.updateAndGet(v -> v + loss);
                    
                    Tensor predictions = classifications.get(targetIndex);
                    int pred = (int) predictions.get(predIndex);
                    
                    predictions.set(pred + 1, predIndex);
                }
            }
        }
    }
    
    private void initLayers() {
        if (layers.isEmpty()) return;
        
        int length = layers.size();
        Layer prev = layers.getFirst();
        
        for (int i = 1; i < length; i++) {
            Layer layer = layers.get(i);
            
            if (layer.isFrozen()) continue;
            
            prev = layer.connect(prev);
        }
        
        IntStream.range(1, length).parallel().forEach(i -> {
            Layer layer = layers.get(i);
            if (layer.isFrozen()) return;
            
            int input = layers.get(i - 1).size();
            int output = layer.size();
            
            RandomGenerator localRandom = new SplittableRandom(seed + i);
            layer.initWeights(localRandom, input, output);
        });
    }
    
    private void append(
        String pattern,
        StringBuilder builder,
        DecimalFormat format,
        AtomicLong totalWeights,
        AtomicLong totalBiases
    ) {
        for (int i = 0; i < layers.size(); i++) {
            Layer layer = layers.get(i);
            String layerType = layer.getClass().getSimpleName();
            
            int biases = layer.totalBiases();
            int weights = layer.totalWeights();
            
            Tensor weightsTensor = layer.getWeights();
            
            String formatWeights = weights == 0 ? "-" : format.format(weights);
            String shape = weightsTensor == null
                ? "[" + biases + "]"
                : Arrays.toString(weightsTensor.shape());
            
            builder.append(pattern.formatted(i, layerType, formatWeights, shape, layer.getActivation().name()));
            
            totalWeights.addAndGet(weights);
            totalBiases.addAndGet(biases);
        }
    }
}
