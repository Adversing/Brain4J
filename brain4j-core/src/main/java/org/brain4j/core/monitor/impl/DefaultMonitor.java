package org.brain4j.core.monitor.impl;

import org.brain4j.core.Brain4J;
import org.brain4j.core.model.Model;
import org.brain4j.core.monitor.Monitor;
import org.brain4j.core.training.Trainer;
import org.brain4j.core.training.events.*;
import org.brain4j.math.commons.Commons;

import java.util.ArrayList;
import java.util.List;

import static org.brain4j.math.Constants.RESET;

public final class DefaultMonitor implements Monitor {
    
    private final List<Double> times = new ArrayList<>();
    private final int timeWindow;
    
    private double batchStart;
    private int epoch;
    private int totalEpochs;
    
    public DefaultMonitor() {
        this(20);
    }
    
    public DefaultMonitor(int timeWindow) {
        this.timeWindow = timeWindow;
    }
    
    @Override
    public void onEvent(TrainingEvent event) {
        switch (event) {
            case BatchStart ignored -> this.batchStart = System.nanoTime();
            case BatchEnd(Trainer ignored, int batch, int totalBatches) -> batchCompleted(batch, totalBatches);
            case EpochStart(Trainer ignored, int epoch, int total) -> epochStarted(epoch, total);
            default -> {}
        }
    }
    
    private void batchCompleted(int batch, int total) {
        double end = System.nanoTime();
        double took = (end - batchStart) / 1e6;
        
        times.add(took);
        
        if (times.size() > timeWindow) {
            times.removeFirst();
        }
        
        double totalTime = times.stream().reduce(Double::sum).orElse(0.0);
        double average = totalTime / Math.min(batch, timeWindow);
        
        if (Brain4J.logging()) {
            printProgress(batch, total, average);
        }
    }
    
    public void epochStarted(int epoch, int total) {
        this.epoch = epoch;
        this.totalEpochs = total;
    }
    
    private void printProgress(int batch, int totalBatches, double tookMs) {
        String barChar = Commons.HEADER_CHAR;
        
        int progressBarLength = 25;
        
        double percentage = (double) batch / totalBatches;
        double tookInSeconds = tookMs / 1000.0;
        
        String timeStr = Commons.formatDuration(tookInSeconds);
        
        String progressBar = Commons.createProgressBar(percentage, progressBarLength, barChar, RESET + barChar);
        String progress = Commons.renderText(" <green>%s ", progressBar);
        
        String intro = Commons.renderText("Epoch <yellow>%s<white>/<yellow>%s", epoch + 1, totalEpochs);
        String batches = Commons.renderText("<blue>%s<white>/<blue>%s <white>batches", batch, totalBatches);
        String time = Commons.renderText("<gray> [%s/batch]<reset>", timeStr);
        
        String message = intro + progress + batches + time;
        System.out.print("\r" + message);
    }
}
