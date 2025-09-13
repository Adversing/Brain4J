import org.brain4j.core.Brain4J;
import org.brain4j.core.activation.Activations;
import org.brain4j.core.layer.impl.DenseLayer;
import org.brain4j.core.layer.impl.utility.InputLayer;
import org.brain4j.core.loss.impl.BinaryCrossEntropy;
import org.brain4j.core.model.Model;
import org.brain4j.core.model.impl.Sequential;
import org.brain4j.core.training.optimizer.impl.AdamW;
import org.brain4j.core.training.wrappers.EvaluationResult;
import org.brain4j.math.Tensors;
import org.brain4j.math.data.ListDataSource;
import org.brain4j.math.data.Sample;
import org.brain4j.math.tensor.Tensor;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

public class TestModel {

    @Test
    public void xorTest() {
        List<Sample> samples = new ArrayList<>();

        for (int x = 0; x <= 1; x++) {
            for (int y = 0; y <= 1; y++) {
                Tensor input = Tensors.vector(x, y);
                Tensor label = Tensors.vector(x ^ y);

                samples.add(new Sample(input, label));
            }
        }

        ListDataSource dataSource = new ListDataSource(samples, true, 4);
        Brain4J.disableLogging();
        
        Model model = Sequential.of(
            new InputLayer(2),
            new DenseLayer(32, Activations.RELU),
            new DenseLayer(32, Activations.RELU),
            new DenseLayer(1, Activations.SIGMOID)
        );
        model.compile(new BinaryCrossEntropy(), new AdamW(0.01));
        model.fit(dataSource, 500);
        
        EvaluationResult result = model.evaluate(dataSource);
        Assertions.assertTrue(result.loss() < 0.005);
    }
}
