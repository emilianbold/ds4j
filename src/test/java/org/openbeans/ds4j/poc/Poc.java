package org.openbeans.ds4j.poc;

import org.openbeans.ds4j.Metal;

import java.lang.foreign.Arena;
import java.lang.foreign.ValueLayout;

/**
 * Proof of concept: pure Java (via the tiny C-ABI Metal bridge) compiles
 * Metal Shading Language source at runtime and runs compute kernels on
 * the GPU, verifying results and measuring bandwidth against a CPU
 * baseline. Demonstrates everything the inference engine will need:
 * runtime MSL compilation, unified-memory buffers zero-copy visible
 * from Java, pipeline dispatch with buffer offsets, and GPU timing.
 */
public final class Poc {

    static final String MSL = """
        #include <metal_stdlib>
        using namespace metal;

        // Memory-bound sanity kernel: c = a + b.
        kernel void poc_vec_add(const device float* a [[buffer(0)]],
                                const device float* b [[buffer(1)]],
                                device float*       c [[buffer(2)]],
                                constant uint&      n [[buffer(3)]],
                                uint gid [[thread_position_in_grid]]) {
            if (gid < n) c[gid] = a[gid] + b[gid];
        }

        // LLM-decode-style half-precision GEMV: y = W * x.
        // One simdgroup (32 lanes) per row; 8 simdgroups per threadgroup.
        kernel void poc_gemv_f16(const device half* w [[buffer(0)]],
                                 const device half* x [[buffer(1)]],
                                 device float*     y [[buffer(2)]],
                                 constant uint2&   mk [[buffer(3)]],
                                 uint sg   [[simdgroup_index_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]],
                                 uint3 tgp [[threadgroup_position_in_grid]]) {
            const uint row = tgp.x * 8u + sg;
            if (row >= mk.x) return;
            const uint K = mk.y;
            const device half* wr = w + (size_t)row * K;
            float acc = 0.0f;
            for (uint i = lane; i < K; i += 32u)
                acc += (float)wr[i] * (float)x[i];
            acc = simd_sum(acc);
            if (lane == 0) y[row] = acc;
        }
        """;

    public static void main(String[] args)  {
        System.out.println("[poc] compiling Metal library from Java-supplied source...");
        Metal m = Metal.open(MSL, null);
        System.out.println("[poc] device: " + m.deviceName);

        vecAddBenchmark(m);
        gemvBenchmark(m);
        System.out.println("[poc] done.");
    }

    static void vecAddBenchmark(Metal m)  {
        final int N = 1 << 25; // 32M floats
        Metal.Buffer a = m.alloc((long) N * 4);
        Metal.Buffer b = m.alloc((long) N * 4);
        Metal.Buffer c = m.alloc((long) N * 4);
        for (int i = 0; i < N; i++) {
            a.setFloat((long) i * 4, i * 1.0e-3f);
            b.setFloat((long) i * 4, 2.0f);
            c.setFloat((long) i * 4, -1.0f);
        }

        {
            var nSeg = Arena.global().allocate(4);
            nSeg.set(ValueLayout.JAVA_INT, 0, N);
            final int T = 256;
            final int grid = (N + T - 1) / T;

            // correctness
            m.run(bb -> {
                bb.bind(0, a, 0).bind(1, b, 0).bind(2, c, 0)
                  .bindBytes(3, nSeg.address(), 4)
                  .launch("poc_vec_add", grid, 1, 1, T, 1, 1);
            });
            for (int i = 0; i < N; i += 9973) {
                float want = i * 1.0e-3f + 2.0f;
                if (Math.abs(c.getFloat((long) i * 4) - want) > 1e-6)
                    throw new AssertionError("vec_add mismatch at " + i);
            }
            System.out.println("[poc] vec_add correctness: PASS");

            // benchmark (read 2 + write 1 stream)
            final int ITERS = 20;
            double bestGpu = Double.MAX_VALUE, bestWall = Double.MAX_VALUE;
            for (int it = 0; it < ITERS; it++) {
                try (Metal.Batch bb = m.begin()) {
                    bb.bind(0, a, 0).bind(1, b, 0).bind(2, c, 0)
                      .bindBytes(3, nSeg.address(), 4)
                      .launch("poc_vec_add", grid, 1, 1, T, 1, 1);
                    bb.commit();
                    long t0 = System.nanoTime();
                    bb.waitGpu();
                    long t1 = System.nanoTime();
                    bestWall = Math.min(bestWall, (t1 - t0) / 1e6);
                    bestGpu = Math.min(bestGpu, bb.gpuTimeMs());
                }
            }
            double bytes = 3.0 * N * 4;
            System.out.printf("[poc] vec_add: N=%d, wall %.3f ms, gpu %.3f ms -> %.1f GB/s effective%n",
                    N, bestWall, bestGpu, bytes / (bestGpu * 1e6));
        }
    }

    static void gemvBenchmark(Metal m)  {
        final int M = 16384, K = 8192; // W = 256 MiB of fp16
        Metal.Buffer w = m.alloc((long) M * K * 2);
        Metal.Buffer x = m.alloc((long) K * 2);
        Metal.Buffer y = m.alloc((long) M * 4);

        // deterministic pseudo-random weights, replicated in Java for the check
        java.util.Random r = new java.util.Random(42);
        float[] xF = new float[K];
        for (int i = 0; i < K; i++) xF[i] = (r.nextFloat() - 0.5f) * 2;
        for (int i = 0; i < K; i++) x.setF16((long) i * 2, xF[i]);
        float[][] checkRows = new float[8][];
        int[] checkIdx = {0, 1, 511, 512, 4095, 4096, 12287, M - 1};
        java.util.Set<Integer> checkSet = new java.util.HashSet<>();
        for (int i : checkIdx) checkSet.add(i);
        for (int row = 0; row < M; row++) {
            boolean save = checkSet.contains(row);
            float[] saveBuf = save ? new float[K] : null;
            for (int i = 0; i < K; i++) {
                float v = (r.nextFloat() - 0.5f) * 2;
                w.setF16(((long) row * K + i) * 2, v);
                if (save) saveBuf[i] = v;
            }
            if (save) checkRows[java.util.Arrays.binarySearch(checkIdx, row)] = saveBuf;
        }

        {
            var mk = Arena.global().allocate(8);
            mk.set(ValueLayout.JAVA_INT, 0, M);
            mk.set(ValueLayout.JAVA_INT, 4, K);
            final int T = 256;
            final int grid = (M + 7) / 8;

            m.run(bb -> bb.bind(0, w, 0).bind(1, x, 0).bind(2, y, 0)
                    .bindBytes(3, mk.address(), 8)
                    .launch("poc_gemv_f16", grid, 1, 1, T, 1, 1));

            // correctness vs CPU reference (fp16-rounded weights)
            for (int c = 0; c < checkIdx.length; c++) {
                int row = checkIdx[c];
                float acc = 0;
                for (int i = 0; i < K; i++)
                    acc += w.getF16(((long) row * K + i) * 2) * xF[i];
                float got = y.getFloat((long) row * 4);
                if (Math.abs(got - acc) > 0.05 * Math.max(1, Math.abs(acc)))
                    throw new AssertionError("gemv mismatch row " + row + ": " + got + " vs " + acc);
            }
            System.out.println("[poc] gemv_f16 correctness: PASS");

            final int ITERS = 20;
            double bestGpu = Double.MAX_VALUE, bestWall = Double.MAX_VALUE;
            for (int it = 0; it < ITERS; it++) {
                try (Metal.Batch bb = m.begin()) {
                    bb.bind(0, w, 0).bind(1, x, 0).bind(2, y, 0)
                      .bindBytes(3, mk.address(), 8)
                      .launch("poc_gemv_f16", grid, 1, 1, T, 1, 1);
                    bb.commit();
                    long t0 = System.nanoTime();
                    bb.waitGpu();
                    long t1 = System.nanoTime();
                    bestWall = Math.min(bestWall, (t1 - t0) / 1e6);
                    bestGpu = Math.min(bestGpu, bb.gpuTimeMs());
                }
            }
            double bytes = (double) M * K * 2 + K * 2 + (double) M * 4;
            double flops = 2.0 * M * K;
            System.out.printf("[poc] gemv_f16: %dx%d, wall %.3f ms, gpu %.3f ms -> %.1f GB/s, %.1f GFLOP/s%n",
                    M, K, bestWall, bestGpu, bytes / (bestGpu * 1e6), flops / (bestGpu * 1e6));

            // CPU baseline: a slice of the same GEMV in plain Java
            final int ROWS = 512;
            float[] wf = new float[K];
            long t0 = System.nanoTime();
            double sliceSum = 0;
            for (int row = 0; row < ROWS; row++) {
                float acc = 0;
                for (int i = 0; i < K; i++)
                    acc += w.getF16(((long) row * K + i) * 2) * xF[i];
                sliceSum += acc;
            }
            double ms = (System.nanoTime() - t0) / 1e6;
            double estFull = ms * M / (double) ROWS;
            System.out.printf("[poc] cpu baseline (1 thread): %d rows %.2f ms -> full GEMV ~%.0f ms (~%.0fx slower than GPU)%n",
                    ROWS, ms, estFull, estFull / bestGpu);
        }
    }
}
