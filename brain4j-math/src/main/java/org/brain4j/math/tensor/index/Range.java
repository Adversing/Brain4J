package org.brain4j.math.tensor.index;

public class Range {
    private final int start;
    private final int end;
    private final int step;

    public static Range interval(int start, int end) {
        return new Range(start, end);
    }
    
    public static Range all() {
        return new Range(0, Integer.MAX_VALUE, 1);
    }

    public static Range point(int i) {
        return new Range(i, i + 1, 1);
    }

    public Range(int start, int end) {
        this(start, end, 1);
    }
    
    public Range(int start, int end, int step) {
        this.start = start;
        this.end = end;
        this.step = step;
    }
    
    public int start(int dimSize) {
        return start >= 0 ? start : start + dimSize;
    }
    
    public int end(int dimSize) {
        return end >= 0 ? Math.min(end, dimSize) : end + dimSize;
    }
    
    public int step() {
        return step;
    }
    
    public int size(int dimSize) {
        int s = start(dimSize);
        int e = end(dimSize);
        return (e - s + step - 1) / step;
    }
    
    @Override
    public String toString() {
        return start + ":" + end + ":" + step;
    }
}