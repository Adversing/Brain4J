package org.brain4j.common.tensor.matmul.impl;

import org.brain4j.common.tensor.Tensor;
import org.brain4j.common.tensor.matmul.MatmulParameters;
import org.brain4j.common.tensor.matmul.MatmulProvider;

import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveAction;

public class NormalMatmulProvider implements MatmulProvider {

    private static final int WORK_THRESHOLD = 1;
    private static final int COMPLEXITY_THRESHOLD = 65536;

    private static boolean isOverThreshold(int work, int np) {
        return work > WORK_THRESHOLD && work * np > COMPLEXITY_THRESHOLD;
    }

    @Override
    public void multiply(ForkJoinPool pool, Tensor a, Tensor b, Tensor c) {
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

        if (!isOverThreshold(work, np)) {
            matmulBlock(A, B, C, 0, work, a.transposed(), b.transposed(), m, n, p, mn, np, mp, batchA, batchB);
            return;
        }

        ScalarAction action = new ScalarAction(0, work, parameters);
        pool.invoke(action);
    }
    
    private void matmulBlock(
        float[] A, float[] B, float[] C,
        int start, int end,
        boolean aT,
        boolean bT,
        int m, int n, int p,
        int mn, int np, int mp,
        int batchA, int batchB
    ) {
        // someone kill me please
        if (!aT && !bT) {
            matmul_NN(A, B, C, start, end, m, n, p, mn, np, mp, batchA, batchB);
        } else if (aT && !bT) {
            matmul_TN(A, B, C, start, end, m, n, p, mn, np, mp, batchA, batchB);
        } else if (!aT) {
            matmul_NT(A, B, C, start, end, m, n, p, mn, np, mp, batchA, batchB);
        } else {
            matmul_TT(A, B, C, start, end, m, n, p, mn, np, mp, batchA, batchB);
        }
    }
    
    private void matmul_NN(
        float[] A, float[] B, float[] C,
        int start, int end,
        int m, int n, int p,
        int mn, int np, int mp,
        int batchA, int batchB
    ) {
        for (int r = start; r < end; r++) {
            int bi = (batchA == 1 ? 0 : r / m) * mn;
            int bj = (batchB == 1 ? 0 : r / m) * np;
            int ci = (r / m) * mp;
            int i = r % m;
            int rowA = bi + i * n;
            int rowC = ci + i * p;
            
            for (int t = 0; t < n; t++) {
                float aVal = A[rowA + t];
                int rowB = bj + t * p;
                for (int j = 0; j < p; j++) {
                    C[rowC + j] += aVal * B[rowB + j];
                }
            }
        }
    }
    
    private void matmul_TN(
        float[] A, float[] B, float[] C,
        int start, int end,
        int m, int n, int p,
        int mn, int np, int mp,
        int batchA, int batchB
    ) {
        for (int r = start; r < end; r++) {
            int bi = (batchA == 1 ? 0 : r / m) * mn;
            int bj = (batchB == 1 ? 0 : r / m) * np;
            int ci = (r / m) * mp;
            int i = r % m;
            int baseA = bi + i;
            int rowC = ci + i * p;
            
            for (int t = 0; t < n; t++) {
                float aVal = A[baseA + t * m];
                int rowB = bj + t * p;
                for (int j = 0; j < p; j++) {
                    C[rowC + j] += aVal * B[rowB + j];
                }
            }
        }
    }
    
    private void matmul_NT(
        float[] A, float[] B, float[] C,
        int start, int end,
        int m, int n, int p,
        int mn, int np, int mp,
        int batchA, int batchB
    ) {
        for (int r = start; r < end; r++) {
            int bi = (batchA == 1 ? 0 : r / m) * mn;
            int bj = (batchB == 1 ? 0 : r / m) * np;
            int ci = (r / m) * mp;
            int i = r % m;
            int rowA = bi + i * n;
            int rowC = ci + i * p;
            
            for (int t = 0; t < n; t++) {
                float aVal = A[rowA + t];
                int baseB = bj + t;
                for (int j = 0; j < p; j++) {
                    C[rowC + j] += aVal * B[baseB + j * n];
                }
            }
        }
    }
    
    private void matmul_TT(
        float[] A, float[] B, float[] C,
        int start, int end,
        int m, int n, int p,
        int mn, int np, int mp,
        int batchA, int batchB
    ) {
        for (int r = start; r < end; r++) {
            int bi = (batchA == 1 ? 0 : r / m) * mn;
            int bj = (batchB == 1 ? 0 : r / m) * np;
            int ci = (r / m) * mp;
            int i = r % m;
            int baseA = bi + i;
            int rowC = ci + i * p;
            
            for (int t = 0; t < n; t++) {
                float aVal = A[baseA + t*m];
                int baseB = bj + t;
                for (int j = 0; j < p; j++) {
                    C[rowC + j] += aVal * B[baseB + j * n];
                }
            }
        }
    }

    private class ScalarAction extends RecursiveAction {
        private final int start, end;
        private final MatmulParameters parameters;

        private ScalarAction(int start, int end, MatmulParameters parameters) {
            this.start = start;
            this.end = end;
            this.parameters = parameters;
        }

        @Override
        protected void compute() {
            int work = end - start;

            if (isOverThreshold(work, parameters.np())) {
                int mid = (start + end) >>> 1;
                invokeAll(
                    new ScalarAction(start, mid, parameters),
                    new ScalarAction(mid, end, parameters)
                );
            } else {
                matmulBlock(
                    parameters.A(), parameters.B(), parameters.C(),
                    start, end,
                    parameters.aTransposed(),
                    parameters.bTransposed(),
                    parameters.m(), parameters.n(), parameters.p(),
                    parameters.mn(), parameters.np(), parameters.mp(),
                    parameters.batchA(), parameters.batchB()
                );
            }
        }
    }
}
