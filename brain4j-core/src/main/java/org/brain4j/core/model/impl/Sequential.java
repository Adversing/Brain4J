package org.brain4j.core.model.impl;

import org.brain4j.core.layer.impl.convolutional.InputLayer;
import org.brain4j.math.Commons;
import org.brain4j.math.Pair;
import org.brain4j.math.Tensors;
import org.brain4j.math.data.ListDataSource;
import org.brain4j.math.gpu.GpuContext;
import org.brain4j.math.gpu.device.Device;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.impl.GpuTensor;
import org.brain4j.math.tensor.index.Range;
import org.brain4j.core.Brain4J;
import org.brain4j.core.layer.Layer;
import org.brain4j.core.loss.LossFunction;
import org.brain4j.core.loss.impl.BinaryCrossEntropy;
import org.brain4j.core.model.Model;
import org.brain4j.core.training.BackPropagation;
import org.brain4j.core.training.StatesCache;
import org.brain4j.core.training.optimizer.Optimizer;
import org.brain4j.core.training.updater.Updater;
import org.brain4j.core.training.wrappers.EvaluationResult;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.text.DecimalFormat;
import java.util.*;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.BiConsumer;
import java.util.stream.IntStream;

import static org.brain4j.math.constants.Constants.*;

/**
 * Represents a simple feedforward neural network model.
 * <p>
 * Supports multiple layer types, loss functions, optimizers, and training via backpropagation.
 * Provides methods for training (fit), prediction, evaluation, and model summary.
 * </p>
 *
 * @author xEcho1337
 * @since 3.0
 */
public class Sequential extends Layer implements Model {

    private static final Logger mainLogger = LoggerFactory.getLogger(Sequential.class);
    private static final Logger trainingLogger = LoggerFactory.getLogger("training");

    /* Data structures in the model */
    protected List<Layer> layers;
    protected List<Layer> flattened;

    /* The device the model is hosted on */
    protected Device device;

    /* General training parameters */
    protected BackPropagation backPropagation;
    protected Optimizer optimizer;
    protected Updater updater;
    protected LossFunction lossFunction;
    protected long seed;

    /**
     * Constructs a new neural network with the given layers.
     * @param layers the sequence of layers forming the neural network
     */
    public static Sequential of(Layer... layers) {
        return new Sequential(layers);
    }

    protected Sequential(Layer... layers) {
        this.layers = new ArrayList<>(List.of(layers));
        this.flattened = new ArrayList<>();
        this.seed = System.currentTimeMillis();

        if (layers.length != 0) {
            Layer layer = layers[0];

            if (!(layer instanceof InputLayer)) {
                throw new IllegalArgumentException("First layer must be an InputLayer!");
            }
        }

        for (Layer layer : layers) {
            if (layer instanceof Model subModel) {
                flattened.addAll(subModel.flattened());
                continue;
            }

            flattened.add(layer);
        }
    }

    protected void connectLayers() {
        if (layers.isEmpty()) return;

        Layer previous = null;
        int size = flattened.size();

        for (int i = 0; i < size; i++) {
            Layer layer = flattenedAt(i);
            previous = layer.connect(previous);
        }

        int[] inputSizes = new int[size];

        for (int i = 1; i < inputSizes.length; i++) {
            inputSizes[i] = flattenedAt(i - 1).size();
        }

        IntStream.range(0, size).parallel().forEach(i -> {
            Layer layer = flattenedAt(i);

            int input = inputSizes[i];
            int output = layer.size();

            Random localRandom = Random.from(new SplittableRandom(seed + i));
            layer.initWeights(localRandom, input, output);
        });
    }

    protected void makeEvaluation(
        Pair<Tensor[], Tensor[]> batch,
        Map<Integer, Tensor> classifications,
        AtomicReference<Double> totalLoss
    ) {
        Tensor[] inputs = batch.first();
        Tensor[] labels = batch.second();

        Tensor[] outputs = predict(new StatesCache(false), inputs);

        for (Tensor input : inputs) {
            int batchSize = input.shape()[0];

            for (int i = 0; i < outputs.length; i++) {
                Tensor output = outputs[i];
                Tensor label = labels[i];

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

    protected void predictBatch(Pair<Tensor[], Tensor[]> batch, AtomicReference<Double> totalError) {
        Tensor[] inputs = batch.first();
        Tensor[] targets = batch.second();

        Tensor[] outputs = predict(new StatesCache(false), inputs);

        for (int i = 0; i < outputs.length; i++) {
            Tensor output = outputs[i].cpu();
            Tensor label = targets[i].cpu();

            int batchSize = output.shape()[0];

            for (int b = 0; b < batchSize; b++) {
                Range range = new Range(b, b + 1);

                Tensor sampleOutput = output.slice(range).flatten();
                Tensor sampleLabel = label.slice(range).flatten();

                double loss = lossFunction.calculate(sampleLabel, sampleOutput);
                totalError.updateAndGet(v -> v + loss);
            }
        }
    }

    protected Tensor[] validateInputs(Tensor... inputs) {
        Tensor[] result = new Tensor[inputs.length];

        for (int i = 0; i < result.length; i++) {
            Tensor input = inputs[i];

            if (input == null || input.rank() == 0) {
                throw new IllegalArgumentException("Input at " + i + " is either null or has dimension of 0!");
            }

            if (input.rank() < 2) {
                // Shape: [batch_size, input_size]
                input = input.reshape(1, input.elements());
            }

            result[i] = input;
        }

        return result;
    }

    protected void printEvaluation(int step, int epoches, ListDataSource testSource) {
        EvaluationResult result = evaluate(testSource.clone());

        double r2 = result.loss() / result.totalDeviation();
        boolean regression = lossFunction.isRegression();

        String lossMsg = "Loss: " + MAGENTA + "%.4f" + RESET;
        String firstMetric = regression
            ? " | R^2 Score: " + LIGHT_BLUE + "%.2f" + RESET
            : " | Accuracy: " + LIGHT_BLUE + "%.2f%%" + RESET;

        String secondMetric = regression ? "" : " | F1-Score: " + LIGHT_GREEN + "%.2f%%" + RESET;

        String prefix = "Epoch " + LIGHT_YELLOW + "%s" + WHITE + "/" + LIGHT_YELLOW + "%s " + WHITE;
        String message = prefix + lossMsg + firstMetric + secondMetric + "\n";
        String formatted = message.formatted(step, epoches,
            result.loss(),
            regression ? r2 : result.accuracy() * 100,
            result.f1Score() * 100
        );

        trainingLogger.info(formatted);
    }

    protected void printProgress(ListDataSource source, int epoch, int epoches, int batch, double tookMs) {
        String barChar = Commons.getHeaderChar();

        int progressBarLength = 25;
        int total = source.batches();

        double percentage = (double) batch / total;
        double tookInSeconds = tookMs / 1000;

        String timeStr = Commons.formatDuration(tookInSeconds);

        String intro = "Epoch " + LIGHT_YELLOW + "%s" + WHITE + "/" + LIGHT_YELLOW + "%s";
        String batchesMsg = LIGHT_BLUE + "%s" + WHITE + "/" + LIGHT_BLUE + "%s " + WHITE + "batches";
        String time = GRAY + " [%s/batch]" + RESET;

        String progressBar = " " + LIGHT_GREEN + Commons.createProgressBar(
            percentage,
            progressBarLength,
            barChar,
            RESET + barChar
        ) + " ";

        String message = String.format(intro + progressBar + batchesMsg + time,
            epoch, epoches, batch, total, timeStr);

        trainingLogger.info(message);
    }

    @Override
    public Model add(Layer layer) {
        layers.add(layer);
        flattened.add(layer);
        return this;
    }

    @Override
    public Model add(int index, Layer layer) {
        layers.add(index, layer);
        flattened.add(index, layer);
        return this;
    }
    
    @Override
    public void fit(ListDataSource train, ListDataSource validation, int epoches, int evaluateEvery) {
        for (int epoch = 1; epoch <= epoches; epoch++) {
            int finalEpoch = epoch;

            List<Double> times = new ArrayList<>();
            AtomicReference<Double> totalForBatch = new AtomicReference<>(0.0);
            
            BiConsumer<Integer, Double> consumer = (batch, took) -> {
                times.add(took);
                totalForBatch.set(totalForBatch.get() + took);

                while (times.size() > 20) {
                    totalForBatch.set(totalForBatch.get() - times.getFirst());
                    times.removeFirst();
                }

                double average = totalForBatch.get() / Math.min(batch, 20);

                if (Brain4J.logging()) {
                    printProgress(train, finalEpoch, epoches, batch, average);
                }
            };
            
            backPropagation.iteration(train, consumer);

            if (epoch % evaluateEvery == 0) {
                if (Brain4J.logging()) {
                    System.out.println();
                }
                
                printEvaluation(epoch, epoches, validation);
            }
        }
    }
    
    @Override
    public Tensor[] predict(StatesCache cache, Tensor... inputs) {
        Tensor[] validated = validateInputs(inputs);
        Tensor[] result = new Tensor[validated.length];

        for (int i = 0; i < validated.length; i++) {
            result[i] = validated[i].to(device).withGrad();
        }

        if (device != null) {
            GpuContext.updateQueue(device, cache.commandQueue());
        }
        
        for (Layer layer : flattened) {
            result = layer.forward(cache, result);
        }

        if (!cache.training() && device != null) {
            GpuContext.closeQueue(device);
        }

        return result;
    }

    @Override
    public void backpropagate(StatesCache cache, Tensor[] outputs, Tensor[] targets) {
        int count = flattened.size() - 1;
        
        Layer last = flattened.getLast();
        last.computeLoss(cache, targets, outputs, lossFunction);

        for (int l = count; l >= 0; l--) {
            Layer layer = flattened.get(l);

            layer.backward(cache, updater, optimizer);
        }
    }

    @Override
    public EvaluationResult evaluate(ListDataSource dataSource) {
        int classes = Math.max(2, dataSource.samples().getFirst().label().elements());
        Map<Integer, Tensor> classifications = new HashMap<>();

        for (int i = 0; i < classes; i++) {
            classifications.put(i, Tensors.zeros(classes));
        }

        AtomicReference<Double> totalLoss = new AtomicReference<>(0.0);
        
        dataSource.reset();
        
        while (dataSource.hasNext()) {
            Pair<Tensor[], Tensor[]> batch = dataSource.nextBatch();
            makeEvaluation(batch, classifications, totalLoss);
        }

        return new EvaluationResult(totalLoss.get() / dataSource.size(), classes, classifications);
    }

    @Override
    public double loss(ListDataSource dataSource) {
        AtomicReference<Double> totalError = new AtomicReference<>(0.0);

        dataSource.reset();

        while (dataSource.hasNext()) {
            Pair<Tensor[], Tensor[]> batch = dataSource.nextBatch();
            predictBatch(batch, totalError);
        }

        return totalError.get() / dataSource.size();
    }
    
    @Override
    public Model to(Device device) {
        this.device = device;
        
        if (device != null) {
            Brain4J.initKernels(device);
        }

        for (Layer layer : flattened) {
            layer.toDevice(device);
        }

        return this;
    }

    @Override
    public Device device() {
        return device;
    }

    @Override
    public Model compile(LossFunction lossFunction, Optimizer optimizer, Updater updater) {
        this.optimizer = optimizer;
        this.updater = updater;
        this.lossFunction = lossFunction;
        this.backPropagation = new BackPropagation(this, optimizer, updater);
        
        updater.resetGradients();
        optimizer.initialize();
        
        zeroGrad();
        connectLayers();
        return this;
    }

    @Override
    public void summary() {
        if (updater == null || optimizer == null) {
            throw new IllegalStateException("The model is not compiled! Make sure to call compile() before.");
        }

        StringBuilder stats = new StringBuilder();
        DecimalFormat format = new DecimalFormat("#,###");

        String pattern = "%-7s %-20s %-12s %-15s %-15s\n";
        String divider = Commons.getHeader(" Architecture ", Commons.getHeaderChar());

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

        stats.append(Commons.getHeader(" Recap ", Commons.getHeaderChar()));
        stats.append("Total weights: %s (%s)\n".formatted(weights, sizeOfWeights));
        stats.append("Total biases: %s (%s)\n".formatted(biases, sizeOfBiases));
        stats.append("Total parameters: %s (%s)\n".formatted(parameters, sizeOfParams));
        stats.append(Commons.getHeader("", Commons.getHeaderChar()));

        Arrays.stream(stats.toString().split("\n")).forEach(mainLogger::info);
    }

    private void append(
        String pattern,
        StringBuilder builder,
        DecimalFormat format,
        AtomicLong totalWeights,
        AtomicLong totalBiases
    ) {
        for (int i = 0; i < flattened.size(); i++) {
            Layer layer = flattenedAt(i);
            String layerType = layer.getClass().getSimpleName();

            int neurons = layer.size();
            int weights = layer.totalWeights() + layer.totalBiases();

            Tensor weightsTensor = layer.weights();

            String formatWeights = weights == 0 ? "-" : format.format(weights);
            String shape = weightsTensor == null
                    ? "[" + neurons + "]"
                    : Arrays.toString(weightsTensor.shape());

            builder.append(pattern.formatted(i, layerType, formatWeights, shape, layer.activation().name()));

            totalWeights.addAndGet(weights);
            totalBiases.addAndGet(neurons);
        }
    }

    @Override
    public Layer layerAt(int index) {
        return layers.get(index);
    }

    @Override
    public Layer flattenedAt(int index) {
        return flattened.get(index);
    }

    @Override
    public void zeroGrad() {
        for (Layer layer : flattened) {
            layer.resetGrad();
        }
    }

    @Override
    public List<Layer> layers() {
        return new ArrayList<>(layers);
    }

    @Override
    public List<Layer> flattened() {
        return new ArrayList<>(flattened);
    }

    @Override
    public Optimizer optimizer() {
        return optimizer;
    }
    
    @Override
    public void setOptimizer(Optimizer optimizer) {
        this.optimizer = optimizer;
    }
    
    @Override
    public Updater updater() {
        return updater;
    }
    
    @Override
    public void setUpdater(Updater updater) {
        this.updater = updater;
    }
    
    @Override
    public LossFunction lossFunction() {
        return lossFunction;
    }
    
    @Override
    public void setLossFunction(LossFunction lossFunction) {
        this.lossFunction = lossFunction;
    }
    
    /**
     * Returns the seed value used to initialize the random number generator.
     * @return the seed value
     */
    public long seed() {
        return seed;
    }

    /**
     * Updates the seed value used to initialize the random number generator.
     * @param seed the new seed value
     * @return the model instance
     */
    public Sequential seed(long seed) {
        this.seed = seed;
        return this;
    }

    @Override
    public Iterator<Layer> iterator() {
        return new Iterator<>() {
            private int currentIndex = 0;

            @Override
            public boolean hasNext() {
                return currentIndex < flattened.size();
            }

            @Override
            public Layer next() {
                return flattenedAt(currentIndex++);
            }
        };
    }

    @Override
    public Layer connect(Layer previous) {
        int size = layers.size();

        for (int i = 0; i < size; i++) {
            Layer layer = layerAt(i);
            previous = layer.connect(previous);
        }

        int[] inputSizes = new int[size];

        for (int i = 0; i < size; i++) {
            inputSizes[i] = (i == 0) ? 0 : layerAt(i - 1).size();
        }

        IntStream.range(0, size).parallel().forEach(i -> {
            Layer layer = layerAt(i);

            int input = inputSizes[i];
            int output = layer.size();

            Random localRandom = Random.from(new SplittableRandom(seed + i));
            layer.initWeights(localRandom, input, output);
        });

        return previous;
    }
    
    @Override
    public Tensor[] forward(StatesCache cache, Tensor... inputs) {
        Tensor[] pass = inputs;
        
        for (int i = 0; i < layers.size(); i++) {
            Layer layer = layerAt(i);

            if (layer == null) {
                throw new IllegalStateException("Layer at index " + i + " is null!");
            }

            pass = layer.forward(cache, pass);
        }

        return pass;
    }

    @Override
    public void backward(StatesCache cache, Updater updater, Optimizer optimizer) {
        for (int l = layers.size() - 2; l >= 0; l--) {
            Layer layer = layerAt(l);

            layer.backward(cache, updater, optimizer);
        }
    }

    @Override
    public int size() {
        return layers.getLast().size();
    }
}
