package org.brain4j.math.scaler.impl;

import com.google.gson.JsonObject;
import org.brain4j.math.Tensors;
import org.brain4j.math.scaler.FeatureScaler;
import org.brain4j.math.tensor.Tensor;

import java.util.List;

public class MinMaxScaler implements FeatureScaler {

    private float rangeMin;
    private float rangeMax;
    private float dataMin;
    private float dataMax;

    private MinMaxScaler() {
    }

    public MinMaxScaler(float rangeMin, float rangeMax) {
        this.rangeMin = rangeMin;
        this.rangeMax = rangeMax;
    }

    @Override
    public void fit(List<Tensor> tensors) {
        float min = Float.POSITIVE_INFINITY;
        float max = Float.NEGATIVE_INFINITY;

        for (Tensor tensor : tensors) {
            float[] data = tensor.data();
            for (float v : data) {
                if (v < min) min = v;
                if (v > max) max = v;
            }
        }

        this.dataMin = min;
        this.dataMax = max;
    }

    @Override
    public Tensor transform(Tensor tensor) {
        float[] data = tensor.data();
        float[] out = new float[data.length];

        for (int i = 0; i < data.length; i++) {
            out[i] = (data[i] - dataMin) / (dataMax - dataMin)
                * (rangeMax - rangeMin) + rangeMin;
        }

        return Tensors.create(tensor.shape(), out);
    }

    @Override
    public void serialize(JsonObject object) {
        object.addProperty("range_min", rangeMin);
        object.addProperty("range_max", rangeMax);
        object.addProperty("data_min", dataMin);
        object.addProperty("data_max", dataMax);
    }

    @Override
    public void deserialize(JsonObject object) {
        this.rangeMin = object.get("range_min").getAsFloat();
        this.rangeMax = object.get("range_max").getAsFloat();
        this.dataMin = object.get("data_min").getAsFloat();
        this.dataMax = object.get("data_max").getAsFloat();
    }

    public float getRangeMin() {
        return rangeMin;
    }

    public void setRangeMin(float rangeMin) {
        this.rangeMin = rangeMin;
    }

    public float getRangeMax() {
        return rangeMax;
    }

    public void setRangeMax(float rangeMax) {
        this.rangeMax = rangeMax;
    }
}