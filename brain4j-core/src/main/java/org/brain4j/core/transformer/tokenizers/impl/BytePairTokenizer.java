package org.brain4j.core.transformer.tokenizers.impl;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonObject;
import com.google.gson.reflect.TypeToken;
import org.brain4j.core.transformer.tokenizers.Tokenizer;
import org.brain4j.math.commons.Commons;
import org.brain4j.math.Tensors;
import org.brain4j.math.tensor.Tensor;

import java.io.*;
import java.lang.reflect.Type;
import java.util.*;
import java.util.concurrent.Callable;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ForkJoinPool;

import static org.brain4j.math.Constants.*;

public class BytePairTokenizer implements Tokenizer {
    
    public static Gson GSON = new GsonBuilder().setPrettyPrinting().create();

    private Map<String, Integer> vocab;
    private Map<String, String[]> merges;
    private String tokenStarter;
    private int bosTokenId;
    private int eosTokenId;
    
    public BytePairTokenizer() {
        this(null);
    }
    
    public BytePairTokenizer(String tokenStarter) {
        this.vocab = new LinkedHashMap<>();
        this.merges = new LinkedHashMap<>();
        this.tokenStarter = tokenStarter;
    }

    @Override
    public List<String> splitTokens(String input) {
        List<String> output = new ArrayList<>();

        for (String word : input.split("\\s+")) {
            boolean hasCharBeforeWord = input.indexOf(word) != 0;
            if (hasCharBeforeWord) word = tokenStarter + word;
            output.addAll(encodeWord(word));
        }

        return output;
    }

    @Override
    public String decode(int index) {
        Optional<Map.Entry<String, Integer>> token = vocab.entrySet()
                .stream()
                .filter(x -> x.getValue() == index)
                .findFirst();

        return token.map(entry -> entry.getKey().replace(tokenStarter, " ")).orElse("<|unk|>");

    }

    @Override
    public Tensor encode(List<String> tokens) {
        Tensor result = Tensors.zeros(tokens.size());

        for (int i = 0; i < tokens.size(); i++) {
            int index = vocab.get(tokens.get(i));
            result.set(index, i);
        }
        
        return result;
    }
    
    @Override
    public int vocabSize() {
        return vocab.size();
    }
    
    @Override
    public int bosTokenId() {
        return bosTokenId;
    }
    
    @Override
    public void setBosTokenId(int bosTokenId) {
        this.bosTokenId = bosTokenId;
    }
    
    @Override
    public int eosTokenId() {
        return eosTokenId;
    }
    
    @Override
    public void setEosTokenId(int eosTokenId) {
        this.eosTokenId = eosTokenId;
    }

    @Override
    public void save(File file) throws IOException {
        if (!file.exists() && !file.getParentFile().mkdirs()) {
            throw new IOException("Cannot create directory: " + file);
        }
        
        JsonObject root = new JsonObject();
        root.addProperty("version", "1.0");
        root.add("truncation", null);
        root.add("padding", null);
        root.add("added_tokens", GSON.toJsonTree(Collections.emptyList()));
        root.add("normalizer", null);
        
        JsonObject preTokenizer = new JsonObject();
        preTokenizer.addProperty("type", "ByteLevel");
        preTokenizer.addProperty("add_prefix_space", false);
        root.add("pre_tokenizer", preTokenizer);
        
        JsonObject postProcessor = new JsonObject();
        postProcessor.addProperty("type", "ByteLevel");
        postProcessor.addProperty("trim_offsets", true);
        root.add("post_processor", postProcessor);
        
        JsonObject decoder = new JsonObject();
        decoder.addProperty("type", "ByteLevel");
        decoder.addProperty("add_prefix_space", false);
        root.add("decoder", decoder);
        
        JsonObject model = new JsonObject();
        model.addProperty("type", "BPE");
        model.add("vocab", GSON.toJsonTree(vocab)); // Map<String, Integer>
        
        List<String> mergeStrings = new ArrayList<>();
        
        for (String[] pair : merges.values()) {
            if (pair.length != 2) continue;
            
            mergeStrings.add(pair[0] + " " + pair[1]);
        }
        
        model.add("merges", GSON.toJsonTree(mergeStrings));
        
        model.add("dropout", null);
        model.add("unk_token", null);
        
        root.add("model", model);
        
        try (Writer writer = new FileWriter(file)) {
            GSON.toJson(root, writer);
        }
    }

    @Override
    public void load(File file) throws IOException {
        if (!file.exists()) throw new FileNotFoundException(file.getPath());
        
        try (Reader reader = new FileReader(file)) {
            JsonObject root = GSON.fromJson(reader, JsonObject.class);
            JsonObject model = root.getAsJsonObject("model");
            JsonObject preTokenizer = root.getAsJsonObject("pre_tokenizer");
            
            String type = preTokenizer.get("type").getAsString();
            
            this.tokenStarter = switch (type) {
                case "ByteLevel" -> "Ġ";
                case "SentencePiece" -> "▁";
                default -> "";
            };
            
            // vocab
            Type vocabType = new TypeToken<LinkedHashMap<String, Integer>>() {}.getType();
            this.vocab = GSON.fromJson(model.getAsJsonObject("vocab"), vocabType);
            
            // merges
            List<String> mergeStrings = GSON.fromJson(
                model.getAsJsonArray("merges"),
                new TypeToken<List<String>>() {}.getType()
            );
            
            LinkedHashMap<String, String[]> loaded = new LinkedHashMap<>();
            
            for (String merge : mergeStrings) {
                String[] pair = merge.split(" ");
                
                if (pair.length == 2) {
                    loaded.put(pair[0] + pair[1], pair);
                }
            }
            
            this.merges = loaded;
        }
    }

    public void fit(List<String> corpus, int numMerges, int evaluateDelay) throws InterruptedException {
        if (merges.isEmpty()) {
            for (String word : corpus) {
                String token = String.join(" ", word.split(""));
                String[] symbols = token.split("\\s+");

                merges.put(token, symbols);
            }
        }

        for (int iter = 0; iter < numMerges; iter++) {
            long start = System.nanoTime();

            Map<String, Integer> pairCounts = new ConcurrentHashMap<>();
            List<Callable<Void>> tasks = new ArrayList<>();

            for (String[] symbols : merges.values()) {
                Callable<Void> task = () -> {
                    for (int i = 0; i < symbols.length - 1; i++) {
                        String pair = symbols[i] + symbols[i + 1];
                        pairCounts.merge(pair, 1, Integer::sum);
                    }
                    return null;
                };

                tasks.add(task);
            }
            
            ForkJoinPool.commonPool().invokeAll(tasks);

            if (pairCounts.isEmpty()) break;

            String bestPair = Collections.max(pairCounts.entrySet(), Map.Entry.comparingByValue()).getKey();
            vocab.put(bestPair, iter);

            Map<String, String[]> updated = new LinkedHashMap<>();

            for (Map.Entry<String, String[]> entry : merges.entrySet()) {
                String[] symbols = entry.getValue();
                List<String> merged = new ArrayList<>();

                for (int i = 0; i < symbols.length; ) {
                    if (i < symbols.length - 1 && (symbols[i] + symbols[i + 1]).equals(bestPair)) {
                        merged.add(bestPair);
                        i += 2;
                    } else {
                        merged.add(symbols[i]);
                        i++;
                    }
                }

                String key = String.join(" ", merged);
                updated.put(key, merged.toArray(new String[0]));
            }

            merges.clear();
            merges.putAll(updated);

            double took = (System.nanoTime() - start) / 1e6;
            printProgressBar(took, iter, numMerges, evaluateDelay);
        }
    }

    public List<String> encodeWord(String word) {
        List<String> symbols = new ArrayList<>(Arrays.asList(word.split("")));
        symbols.add("</w>");

        while (true) {
            Map<String, Integer> candidates = new HashMap<>();

            for (int i = 0; i < symbols.size() - 1; i++) {
                String pair = symbols.get(i) + symbols.get(i + 1);
                int rank = vocab.getOrDefault(pair, Integer.MAX_VALUE);

                candidates.put(pair, rank);
            }

            String best = null;
            int bestRank = Integer.MAX_VALUE;

            for (Map.Entry<String, Integer> entry : candidates.entrySet()) {
                if (entry.getValue() < bestRank) {
                    best = entry.getKey();
                    bestRank = entry.getValue();
                }
            }

            if (best == null) break;

            symbols = getSymbols(symbols, best);
        }

        if (!symbols.isEmpty() && symbols.get(symbols.size() - 1).equals("</w>")) {
            symbols.remove(symbols.size() - 1);
        }

        return symbols;
    }

    private static List<String> getSymbols(List<String> symbols, String best) {
        List<String> newSymbols = new ArrayList<>();

        for (int i = 0; i < symbols.size(); ) {
            String cur = symbols.get(i);
            String next = (i < symbols.size() - 1) ? symbols.get(i + 1) : null;

            if (next != null && (cur + next).equals(best)) {
                newSymbols.add(best);
                i += 2;
            } else {
                newSymbols.add(cur);
                i++;
            }
        }

        return newSymbols;
    }

    private void printProgressBar(
        double tookMs,
        int iteration,
        int merges,
        int evaluateDelay
    ) {
        int progressBarLength = 20;
        double percentage = (double) iteration / merges;

        String barChar = Commons.getHeaderChar();
        int remaining = merges - iteration;

        double seconds = tookMs / 1000;
        double remainingTime = seconds * remaining;

        String remainingTimeStr = Commons.formatDuration(remainingTime);
        String timeStr = Commons.formatDuration(seconds);

        String progressMsg = WHITE + "[%s/%s] ";
        String progressBar = LIGHT_GREEN + Commons.createProgressBar(
                percentage,
                progressBarLength,
                barChar,
                RESET + barChar
        );

        String percentual = LIGHT_YELLOW + " %.2f%%" + RESET;
        String time = GRAY + " [%s/epoch | %s remaining]" + RESET;
        String message = String.format(progressMsg + progressBar + percentual + time,
                iteration, merges, percentage * 100, timeStr, remainingTimeStr);

        System.out.print(message);

        if (iteration % evaluateDelay == 0) {
            printEvaluation(iteration, merges);
        }
    }

    private void printEvaluation(int iteration, int total) {
        System.out.println();

        String symbolsMsg = "Symbols: " + LIGHT_BLUE + "%,d" + RESET;
        String tokensMsg = "Tokens: " + LIGHT_GREEN + "%,d" + RESET;

        String message = "[%s/%s] " + symbolsMsg + " | " + tokensMsg + "\n";
        String formatted = String.format(message, iteration, total, merges.size(), totalSymbols());

        System.out.print(formatted);
    }

    private int totalSymbols() {
        return merges.values()
                .stream()
                .mapToInt(x -> x.length)
                .sum();
    }

    public Map<String, Integer> vocab() {
        return Collections.unmodifiableMap(vocab);
    }
    
    public Map<String, String[]> merges() {
        return Collections.unmodifiableMap(merges);
    }

    public void clearTokens() {
        merges.clear();
    }

    public void clearEncodings() {
        vocab.clear();
    }
}
