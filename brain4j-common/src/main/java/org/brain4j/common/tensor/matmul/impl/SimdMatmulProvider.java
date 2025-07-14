package org.brain4j.common.tensor.matmul.impl;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.VectorSpecies;
import org.brain4j.common.tensor.Tensor;
import org.brain4j.common.tensor.matmul.MatmulParameters;
import org.brain4j.common.tensor.matmul.MatmulProvider;

import java.util.concurrent.ForkJoinTask;
import java.util.concurrent.RecursiveAction;

public class SimdMatmulProvider implements MatmulProvider {

    private static final VectorSpecies<Float> FLOAT_SPECIES = FloatVector.SPECIES_PREFERRED;
    private static final VectorSpecies<Integer> INT_SPECIES = IntVector.SPECIES_PREFERRED;
    private static final int VL = INT_SPECIES.length();

    private static final int PARALLELISM = Runtime.getRuntime().availableProcessors();
    private static final int PARALLEL_COMPLEXITY_THRESHOLD = 65536;
    private static final int PARALLEL_WORK_THRESHOLD = PARALLELISM;

    private static final int SPLIT_COMPLEXITY_THRESHOLD = 65536 * 4;
    private static final int SPLIT_WORK_THRESHOLD = 2;

    private static boolean isOverParallelThreshold(int work, int np) {
        return work > PARALLEL_WORK_THRESHOLD && work * np > PARALLEL_COMPLEXITY_THRESHOLD;
    }
    
    private static boolean isOverSplitThreshold(int work, int np) {
        return work > SPLIT_WORK_THRESHOLD && work * np > SPLIT_COMPLEXITY_THRESHOLD;
    }

    private final int[] strideArray = new int[VL];
    
    @Override
    public void multiply(Tensor a, Tensor b, Tensor c) {
        float[] A = a.data();
        float[] B = b.data();
        float[] C = c.data();
        
        int rankA = a.rank();
        int rankB = b.rank();
        
        int[] shapeA = a.shape();
        int[] shapeB = b.shape();
        int[] shapeC = c.shape();
        
        int batchA = 1;
        int batchB = 1;
        int batch = 1;
        
        for (int i = 0; i < rankA - 2; i++) batchA *= shapeA[i];
        for (int i = 0; i < rankB - 2; i++) batchB *= shapeB[i];
        for (int i = 0; i < shapeC.length - 2; i++) batch *= shapeC[i];
        
        int m = shapeA[rankA - 2];
        int n = shapeA[rankA - 1];
        int p = shapeB[rankB - 1];
        
        MatmulParameters parameters = new MatmulParameters(A, B, C, m, n, p, a.transposed(), b.transposed(), batchA, batchB);
        
        int work = batch * m;
        
        int mn = m * n;
        int np = n * p;
        int mp = m * p;
        
        if (!isOverParallelThreshold(work, np)) {
            matmulBlock(A, B, C, 0, work, m, n, p, mn, np, mp, batchA, batchB, a.transposed(), b.transposed());
            return;
        }
        
        int step = work / PARALLELISM;
        VectorAction[] actions = new VectorAction[PARALLELISM];
        
        for (int i = 0; i < PARALLELISM; i++) {
            int startIndex = i * step;
            int endIndex = Math.min(startIndex + step, work);
            actions[i] = new VectorAction(parameters, startIndex, endIndex);
        }
        
        ForkJoinTask.invokeAll(actions);
    }
    
    private void matmulBlock(
        float[] a, float[] b, float[] c,
        int start, int end,
        int m, int n, int p,
        int mn, int np, int mp,
        int batchA, int batchB,
        boolean transposedA, boolean transposedB
    ) {
        if (batchA == 1 && batchB == 1) {
            matmulSimple(a, b, c, start, end, m, n, p, mn, np, mp, transposedA, transposedB);
        } else {
            matmulGeneric(a, b, c, start, end, m, n, p, mn, np, mp, batchA, batchB, transposedA, transposedB);
        }
    }
    
    private void matmulSimple(
        float[] a, float[] b, float[] c,
        int start, int end,
        int m, int n, int p,
        int mn, int np, int mp,
        boolean transposedA, boolean transposedB
    ) {
        for (int r = start; r < end; r++) {
            int batch = r / m;
            int i = r % m;
            int offsetA = batch * mn;
            int offsetB = batch * np;
            int offsetC = batch * mp;
            int rowA = offsetA + i * n;
            int rowC = offsetC + i * p;
            
            for (int t = 0; t < n; t++) {
                float aVal = a[rowA + t];
                int colB = offsetB + t * p;
                
                int j = 0;

                for (; j < FLOAT_SPECIES.loopBound(p); j += FLOAT_SPECIES.length()) {
                    FloatVector vb = FloatVector.fromArray(FLOAT_SPECIES, b, colB + j);
                    FloatVector vc = FloatVector.fromArray(FLOAT_SPECIES, c, rowC + j);
                    vc.add(vb.mul(aVal)).intoArray(c, rowC + j);
                }
                
                for (; j < p; j++) {
                    c[rowC + j] += aVal * b[colB + j];
                }
            }
        }
    }
    
    private void matmulGeneric(
        float[] a, float[] b, float[] c,
        int start, int end,
        int m, int n, int p,
        int mn, int np, int mp,
        int batchA, int batchB,
        boolean transposedA, boolean transposedB
    ) {
        for (int r = start; r < end; r++) {
            int bi = (batchA == 1 ? 0 : r / m) * mn;
            int bj = (batchB == 1 ? 0 : r / m) * np;
            int ci = (r / m) * mp;
            int i = r % m;
            int rowA = bi + i * n;
            int rowC = ci + i * p;
            
            for (int t = 0; t < n; t++) {
                float aVal = a[rowA + t];
                int colB = bj + t * p;
                
                int j = 0;
                
                for (; j < FLOAT_SPECIES.loopBound(p); j += FLOAT_SPECIES.length()) {
                    FloatVector vb = FloatVector.fromArray(FLOAT_SPECIES, b, colB + j);
                    FloatVector vc = FloatVector.fromArray(FLOAT_SPECIES, c, rowC + j);
                    vc.add(vb.mul(aVal)).intoArray(c, rowC + j);
                }
                
                for (; j < p; j++) {
                    c[rowC + j] += aVal * b[colB + j];
                }
            }
        }
    }
    
    private class VectorAction extends RecursiveAction {

        private final MatmulParameters parameters;
        private final int start;
        private final int end;

        public VectorAction(MatmulParameters parameters, int start, int end) {
            this.parameters = parameters;
            this.start = start;
            this.end = end;
        }

        @Override
        protected void compute() {
            int np = parameters.np();
            int work = end - start;
            
            if (isOverSplitThreshold(work, np)) {
                int mid = (start + end) >>> 1;
                invokeAll(
                    new VectorAction(parameters, start, mid),
                    new VectorAction(parameters, mid, end)
                );
            } else {
                matmulBlock(
                    parameters.A(), parameters.B(), parameters.C(),
                    start, end,
                    parameters.m(), parameters.n(), parameters.p(),
                    parameters.mn(), parameters.np(), parameters.mp(),
                    parameters.batchA(), parameters.batchB(),
                    parameters.transposedA(), parameters.transposedB()
                );
            }
        }
    }
}