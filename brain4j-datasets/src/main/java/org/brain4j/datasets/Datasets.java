package org.brain4j.datasets;

import org.brain4j.datasets.core.dataset.Dataset;
import org.brain4j.datasets.core.loader.DatasetLoader;
import org.brain4j.datasets.core.loader.config.LoadConfig;
import org.brain4j.datasets.download.callback.ProgressCallback;

import java.util.concurrent.CompletableFuture;

public final class Datasets {

    private Datasets() {}

    public static Dataset loadDataset(String datasetId) throws Exception {
        try (DatasetLoader loader = new DatasetLoader()) {
            return loader.loadDataset(datasetId);
        } catch (Exception e) {
            throw new Exception("Failed to load dataset: " + datasetId, e);
        }
    }

    public static Dataset loadDataset(String datasetId, LoadConfig config) throws Exception {
        try (DatasetLoader loader = new DatasetLoader()) {
            return loader.loadDataset(datasetId, config);
        } catch (Exception e) {
            throw new Exception("Failed to load dataset: " + datasetId, e);
        }
    }

    public static Dataset loadDataset(String datasetId, ProgressCallback progressCallback) throws Exception {
        try (DatasetLoader loader = new DatasetLoader(progressCallback)) {
            return loader.loadDataset(datasetId);
        } catch (Exception e) {
            throw new Exception("Failed to load dataset: " + datasetId, e);
        }
    }

    public static CompletableFuture<Dataset> loadDatasetAsync(String datasetId) {
        try (DatasetLoader loader = new DatasetLoader()) {
            return loader.loadDatasetAsync(datasetId);
        } catch (Exception e) {
            return CompletableFuture.failedFuture(e);
        }
    }

    public static CompletableFuture<Dataset> loadDatasetAsync(String datasetId, LoadConfig config) {
        try (DatasetLoader loader = new DatasetLoader()) {
            return loader.loadDatasetAsync(datasetId, config);
        } catch (Exception e) {
            return CompletableFuture.failedFuture(e);
        }
    }
}