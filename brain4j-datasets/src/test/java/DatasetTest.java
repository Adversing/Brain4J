import org.brain4j.datasets.Datasets;
import org.brain4j.datasets.core.dataset.Dataset;
import org.brain4j.datasets.format.impl.ParquetFormat;
import org.brain4j.math.data.ListDataSource;

public class DatasetTest {
    public static void main(String[] args) throws Exception {
        Dataset dataset = Datasets.loadDataset("imdb");
        ListDataSource dataSource = dataset.createDataSource(
            new ParquetFormat(),
            (record, lineIndex) -> {
                int label = Integer.parseInt(String.valueOf(record.get("label")));
                String text = String.valueOf(record.get("text"));
                
                System.out.println("Label: " + label);
                return null;
            },
        false, 16);
    }
}
