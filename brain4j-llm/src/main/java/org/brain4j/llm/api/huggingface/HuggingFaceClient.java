package org.brain4j.llm.api.huggingface;

import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.apache.hc.client5.http.classic.methods.HttpGet;
import org.apache.hc.client5.http.impl.classic.CloseableHttpClient;
import org.apache.hc.client5.http.impl.classic.CloseableHttpResponse;
import org.apache.hc.client5.http.impl.classic.HttpClients;
import org.apache.hc.core5.http.ParseException;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.brain4j.llm.api.ModelInfo;
import org.brain4j.math.commons.result.Result;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.net.URI;
import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;

public class HuggingFaceClient implements AutoCloseable {

    private static final Logger logger = LoggerFactory.getLogger(HuggingFaceClient.class);
    private static final String BASE_URL = "https://huggingface.co";
    private static final String API_BASE_URL = BASE_URL + "/api/models/";

    private final CloseableHttpClient httpClient;
    private final ObjectMapper objectMapper;
    private final String userAgent = "brain4j-llm/1.0.0";

    public HuggingFaceClient() {
        this.httpClient = HttpClients.createDefault();
        this.objectMapper = new ObjectMapper()
            .configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);
    }

    public Result<ModelInfo, HuggingFaceException> getModelInfo(String modelId) {
        validateId(modelId);

        String encoded = URLEncoder.encode(modelId, StandardCharsets.UTF_8);
        String url = API_BASE_URL + encoded;

        HttpGet request = new HttpGet(URI.create(url));
        request.setHeader("User-Agent", userAgent);

        try (CloseableHttpResponse res = httpClient.execute(request)) {
            int code = res.getCode();
            String body = EntityUtils.toString(res.getEntity());

            return switch (code) {
                case 200 -> {
                    ModelInfo info = objectMapper.readValue(body, ModelInfo.class);
                    yield Result.ok(info);
                }
                case 404 -> {
                    logger.warn("Model not found or private: {}", modelId);
                    yield Result.err(new HuggingFaceException("Model not found or private: " + modelId));
                }
                default -> Result.err(new HuggingFaceException("Failed to get model info (" + code + "): " + body));
            };
        } catch (IOException | ParseException e) {
            return Result.err(new HuggingFaceException("Network or parse error while fetching " + modelId, e));
        }
    }

    public FileDownloadResponse downloadFile(String modelId, String filename) throws HuggingFaceException {
        validateId(modelId);
        validateFilename(filename);

        String encodedId = URLEncoder.encode(modelId, StandardCharsets.UTF_8);
        String encodedFile = URLEncoder.encode(filename, StandardCharsets.UTF_8);
        String url = BASE_URL + "/" + encodedId + "/resolve/main/" + encodedFile;

        try {
            HttpGet req = new HttpGet(URI.create(url));
            req.setHeader("User-Agent", userAgent);

            CloseableHttpResponse response = httpClient.execute(req);

            if (response.getCode() != 200) {
                String body = EntityUtils.toString(response.getEntity());
                throw new HuggingFaceException("Download failed (" + response.getCode() + "): " + body);
            }

            return new FileDownloadResponse(response, response.getEntity().getContent());
        } catch (IOException | ParseException e) {
            throw new HuggingFaceException("I/O error while downloading " + filename, e);
        }
    }

    private void validateId(String id) {
        if (id == null || id.isBlank()) {
            throw new IllegalArgumentException("Model ID cannot be null or empty");
        }
    }

    private void validateFilename(String name) {
        if (name == null || name.isBlank()) {
            throw new IllegalArgumentException("Filename cannot be null or empty");
        }
    }

    @Override
    public void close() throws IOException {
        httpClient.close();
    }
}