package org.brain4j.core.layer.impl.utility;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.index.Range;
import org.brain4j.core.layer.Layer;
import org.brain4j.core.training.StatesCache;

import java.util.List;

public class SliceLayer extends Layer {
    
    private Range[] ranges;
    private int size;
    
    public SliceLayer(Range... ranges) {
        this.ranges = ranges;
    }
    
    @Override
    public Layer connect(Layer previous) {
        this.size = previous.size();
        return this;
    }
    
    @Override
    public Tensor[] forward(StatesCache cache, Tensor... inputs) {
        Tensor[] result = new Tensor[inputs.length];

        for (int i = 0; i < inputs.length; i++) {
            result[i] = inputs[i].sliceGrad(ranges);
        }

        return result;
    }
    
    @Override
    public int size() {
        return size;
    }
    
    @Override
    public void serialize(JsonObject object) {
        JsonArray rangesArray = new JsonArray();
        
        for (Range range : ranges) {
            JsonArray array = new JsonArray();
            array.add(range.start());
            array.add(range.end());
            array.add(range.step());
            rangesArray.add(array);
        }
        
        object.addProperty("size", size);
        object.add("ranges", rangesArray);
    }
    
    @Override
    public void deserialize(JsonObject object) {
        JsonArray rangesArray = object.get("ranges").getAsJsonArray();

        this.size = object.get("size").getAsInt();
        this.ranges = new Range[rangesArray.size()];
        
        for (int i = 0; i < ranges.length; i++) {
            JsonArray range = rangesArray.get(i).getAsJsonArray();
            int start = range.get(0).getAsInt();
            int end = range.get(1).getAsInt();
            int step = range.get(2).getAsInt();
            this.ranges[i] = new Range(start, end, step);
        }
    }
}
