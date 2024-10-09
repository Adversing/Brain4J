package net.echo.brain4j.data;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class DataSet {

    private final List<DataRow> dataRows;

    public DataSet(DataRow... rows) {
        this.dataRows = new ArrayList<>(Arrays.asList(rows));
    }

    public List<DataRow> getDataRows() {
        return dataRows;
    }
}