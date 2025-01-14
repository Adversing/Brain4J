import net.echo.brain4j.activation.Activations;
import net.echo.brain4j.layer.impl.DenseLayer;
import net.echo.brain4j.loss.LossFunctions;
import net.echo.brain4j.model.Model;
import net.echo.brain4j.model.initialization.WeightInit;
import net.echo.brain4j.training.data.DataRow;
import net.echo.brain4j.training.data.DataSet;
import net.echo.brain4j.training.optimizers.impl.AdamGPU;
import net.echo.brain4j.training.optimizers.impl.AdamW;
import net.echo.brain4j.training.optimizers.impl.AdamWGPU;
import net.echo.brain4j.training.updater.impl.NormalUpdater;
import net.echo.brain4j.utils.Vector;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import javax.swing.*;
import java.awt.*;
import java.util.function.Function;

public class PolynomTest {

    public static final Function<Double, Double> FUNCTION = x -> Math.pow(x, 4) - 5 * Math.pow(x, 2) - x + 10;
    public static final double BOUND = 3;

    private final XYSeries trueSeries;
    private final XYSeries predictedSeries;
    private final JLabel errorLabel;

    public PolynomTest() {
        this.trueSeries = new XYSeries("True Output");
        this.predictedSeries = new XYSeries("Predicted Output");
        this.errorLabel = new JLabel("Error: 0.0");
    }

    public static void main(String[] args) {
        PolynomTest app = new PolynomTest();
        app.trainAndVisualize();
    }

    public Model getModel() {
        Model model = new Model(
                new DenseLayer(1, Activations.LINEAR),
                new DenseLayer(64, Activations.SIGMOID),
                new DenseLayer(32, Activations.SIGMOID),
                new DenseLayer(16, Activations.SIGMOID),
                new DenseLayer(1, Activations.SIGMOID)
        );

        model.compile(
                WeightInit.UNIFORM_XAVIER,
                LossFunctions.MEAN_SQUARED_ERROR,
                new AdamGPU(0.01),
                new NormalUpdater()
        );

        return model;
    }

    public DataSet getDataSet() {
        DataSet set = new DataSet();

        double yBound = FUNCTION.apply(BOUND);

        for (int i = 0; i < 500; i++) {
            double x = Math.random() * BOUND * 2 - BOUND;
            double y = FUNCTION.apply(x) / yBound;

            set.getData().add(new DataRow(Vector.of(x), Vector.of(y)));
        }

        set.shuffle();
        set.partition(32);

        return set;
    }

    public JPanel createChartPanel() {
        XYSeriesCollection dataset = new XYSeriesCollection(trueSeries);
        dataset.addSeries(predictedSeries);

        JFreeChart chart = ChartFactory.createXYLineChart(
                "Polynomial Optimization",
                "X Value",
                "Value",
                dataset,
                PlotOrientation.VERTICAL,
                true,
                true,
                false
        );

        ChartPanel panel = new ChartPanel(chart);
        panel.setPreferredSize(new Dimension(800, 600));

        JPanel container = new JPanel();
        container.setLayout(new BorderLayout());
        container.add(panel, BorderLayout.CENTER);
        container.add(errorLabel, BorderLayout.SOUTH);

        return container;
    }

    public void updateGraph(double x, double trueOutput, double predictedOutput) {
        trueSeries.add(x, trueOutput);
        predictedSeries.add(x, predictedOutput);
    }

    public void trainAndVisualize() {
        Model model = getModel();
        DataSet dataSet = getDataSet();

        JFrame frame = new JFrame("Polynomial Optimization");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.add(createChartPanel());
        frame.pack();
        frame.setVisible(true);

        int i = 0;

        long start = System.nanoTime();
        long lastFit = System.nanoTime();

        for (int j = 0; j <= 100; j++) {
            model.fit(dataSet);

            if (j % 100 == 0) {
                double took = (System.nanoTime() - lastFit) / 1e8;
                predictedSeries.clear();
                lastFit = System.nanoTime();

                for (DataRow row : dataSet.getData()) {
                    double x = row.inputs().get(0);
                    double trueOutput = row.outputs().get(0);

                    Vector prediction = model.predict(row.inputs());
                    double predictedOutput = prediction.get(0);

                    updateGraph(x, trueOutput, predictedOutput);
                }

                double error = model.evaluate(dataSet);
                errorLabel.setText("Error: " + String.format("%.4f", error) + " Took: " + String.format("%.2f", took) + " ms Epoch: " + i);
                System.out.println("Epoch #" + j + " Error: " + error + " Took: " + took + " ms");
            }
        }

        double took = (System.nanoTime() - start) / 1e6;
        System.out.println("In total took: " + took + " ms");
    }
}