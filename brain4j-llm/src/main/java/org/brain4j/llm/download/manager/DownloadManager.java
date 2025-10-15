package org.brain4j.llm.download.manager;

import org.brain4j.llm.api.FileDownloadResponse;
import org.brain4j.llm.api.HuggingFaceClient;
import org.brain4j.llm.api.HuggingFaceException;
import org.brain4j.llm.cache.manager.CacheManager;
import org.brain4j.llm.download.callback.ProgressCallback;
import org.brain4j.math.commons.result.Result;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.Executor;
import java.util.concurrent.ForkJoinPool;

public record DownloadManager(HuggingFaceClient client, CacheManager cacheManager, Executor executor,
                              ProgressCallback progressCallback) {

    private static final Logger log = LoggerFactory.getLogger(DownloadManager.class);

    public DownloadManager(HuggingFaceClient client, CacheManager cacheManager) {
        this(client, cacheManager, ForkJoinPool.commonPool(), ProgressCallback.NOOP);
    }

    public Result<Path, Exception> downloadFile(String modelId, String filename) {
        return downloadFile(modelId, filename, false);
    }

    public Result<Path, Exception> downloadFile(String modelId, String filename, boolean force) {
        Path cachedPath = cacheManager.getCachedFilePath(modelId, filename);

        if (!force && cacheManager.isCached(modelId, filename)) {
            log.debug("Using cached file: {}", cachedPath);
            progressCallback.onProgress(filename, 100.0, "Using cached file");
            return Result.ok(cachedPath);
        }

        log.info("Downloading '{}' from model '{}'", filename, modelId);
        progressCallback.onProgress(filename, 0.0, "Starting download");

        try {
            Files.createDirectories(cachedPath.getParent());

            try (FileDownloadResponse response = client.downloadFile(modelId, filename)) {
                log.debug("Saving file to: {}", cachedPath);

                Files.copy(response.inputStream(), cachedPath, StandardCopyOption.REPLACE_EXISTING);

                if (!Files.exists(cachedPath)) {
                    throw new IOException("File was not created successfully: " + cachedPath);
                }

                long fileSize = Files.size(cachedPath);
                log.debug("File saved successfully ({} bytes)", fileSize);

                progressCallback.onProgress(filename, 100.0, "Download complete");
                log.info("Successfully downloaded: {}", cachedPath);
                return Result.ok(cachedPath);
            } catch (HuggingFaceException e) {
                return Result.err(e);
            }
        } catch (IOException e) {
            try {
                Files.deleteIfExists(cachedPath);
            } catch (IOException ex) {
                return Result.err(ex);
            }
            return Result.err(e);
        }
    }

    public CompletableFuture<Result<Path, Exception>> downloadFileAsync(String modelId, String filename) {
        return downloadFileAsync(modelId, filename, false);
    }

    public CompletableFuture<Result<Path, Exception>> downloadFileAsync(String modelId, String filename, boolean force) {
        return CompletableFuture.supplyAsync(() -> {
            return downloadFile(modelId, filename, force);
        }, executor);
    }
}