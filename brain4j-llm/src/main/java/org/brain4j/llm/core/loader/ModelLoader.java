package org.brain4j.llm.core.loader;

import org.brain4j.core.transformer.tokenizers.Tokenizer;
import org.brain4j.core.transformer.tokenizers.impl.BytePairTokenizer;
import org.brain4j.llm.api.huggingface.HuggingFaceClient;
import org.brain4j.llm.api.ModelInfo;
import org.brain4j.llm.api.ModelInfo.Sibling;
import org.brain4j.llm.cache.manager.CacheManager;
import org.brain4j.llm.core.loader.config.LoadConfig;
import org.brain4j.llm.core.model.LLM;
import org.brain4j.llm.api.ModelFile;
import org.brain4j.llm.download.callback.ProgressCallback;
import org.brain4j.llm.download.manager.DownloadManager;
import org.brain4j.math.commons.result.Result;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.ForkJoinPool;

public class ModelLoader implements AutoCloseable {

    private static final Logger log = LoggerFactory.getLogger(ModelLoader.class);

    private final HuggingFaceClient client;
    private final CacheManager cacheManager;
    private final DownloadManager downloadManager;

    public ModelLoader() {
        this.client = new HuggingFaceClient();
        this.cacheManager = new CacheManager();
        this.downloadManager = new DownloadManager(client, cacheManager);
    }

    public ModelLoader(ProgressCallback progressCallback) {
        this.client = new HuggingFaceClient();
        this.cacheManager = new CacheManager();
        this.downloadManager = new DownloadManager(client, cacheManager, ForkJoinPool.commonPool(), progressCallback);
    }

    public LLM loadModel(String modelId) throws Exception {
        return loadModel(modelId, LoadConfig.defaultConfig());
    }
    
    public Tokenizer loadTokenizer(String tokenizerId) throws Exception {
        return loadTokenizer(tokenizerId, LoadConfig.defaultConfig());
    }
    
    public Tokenizer loadTokenizer(String tokenizerId, LoadConfig config) throws Exception {
        log.info("Loading tokenizer: {}", tokenizerId);
        
        ModelInfo info = client.getModelInfo(tokenizerId).unwrap();
        log.debug("Tokenizer info retrieved for: {} (resolved id: {})", tokenizerId, info.id());
        
        String fileToDownload = "tokenizer.json";
        
        if (info.siblings().stream().noneMatch(x -> x.rfilename().equals(fileToDownload))) {
            throw new FileNotFoundException("File not found: " + fileToDownload);
        }
        
        Result<Path, Exception> filePathResult = downloadManager.downloadFile(info.id(), fileToDownload, config.forceDownload());
        Tokenizer tokenizer = new BytePairTokenizer();
        
        Path path = filePathResult.unwrap();
        tokenizer.load(path.toFile());

        return tokenizer;
    }
    
    public LLM loadModel(String modelId, LoadConfig config) throws Exception {
        log.info("Loading model: {}", modelId);

        ModelInfo info = client.getModelInfo(modelId).unwrap();
        log.debug("Model info retrieved for: {} (resolved id: {})", modelId, info.id());

        List<String> filesToDownload = determineFilesToDownload(info, config);
        log.debug("Files to download: {}", filesToDownload);

        List<ModelFile> files = new ArrayList<>();

        for (String filename : filesToDownload) {
            Result<Path, Exception> filePathResult = downloadManager.downloadFile(info.id(), filename, config.forceDownload());

            Path path = filePathResult.unwrap();
            String format = determineFileFormat(filename);

            long size = Files.size(path);
            files.add(new ModelFile(filename, path, size, format));
            log.debug("Added file: {} ({} bytes, {})", filename, size, format);
        }

        Map<String, Object> metadata = new HashMap<>();
        metadata.put("streaming", config.streaming());
        metadata.put("split", config.split());
        
        LLM model = new LLM(info.id(), info, files, metadata);
        log.info("Successfully loaded model: {} ({} files)", info.id(), files.size());

        return model;
    }

    private List<String> determineFilesToDownload(ModelInfo info, LoadConfig config) {
        List<String> all = new ArrayList<>();

        if (info.siblings() != null) {
            for (Sibling s : info.siblings()) {
                String name = s.rfilename();
                if (isUsefulFile(name)) all.add(name);
            }
        }

        if (config.maxFiles() > 0 && all.size() > config.maxFiles()) {
            return all.subList(0, config.maxFiles());
        }
        return all;
    }

    private boolean isUsefulFile(String filename) {
        if (filename.startsWith(".")) return false;
        return !filename.equalsIgnoreCase("README.md")
            && !filename.equalsIgnoreCase(".gitattributes")
            && !filename.endsWith(".lock")
            && !filename.endsWith(".txt");
    }

    private String determineFileFormat(String filename) {
        if (filename.endsWith(".json")) return "json";
        if (filename.endsWith(".bin")) return "bin";
        if (filename.endsWith(".safetensors")) return "safetensors";
        return "unknown";
    }

    @Override
    public void close() throws IOException {
        client.close();
    }
}