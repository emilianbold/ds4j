package org.openbeans.ds4j.poc;

import org.openbeans.ds4j.*;

/** Unit test for the tl_compressor_pool_ratio4_f32_512 launch path. */
public final class PoolTest {
    public static void main(String[] a) throws Exception {
        Metal m = Metal.open(MetalSources.source(), "DS4_METAL_HAS_TENSOR=1\nDS4_METAL_HC_STABLE=1\nDS4_METAL_NORM_RSQRT_DISABLE=1");
        Tl tl = new Tl(m, System.getProperty("ds4j.tunedDir"));
        Metal.Buffer dst = m.alloc(512 * 4);
        Metal.Buffer kv = m.alloc(8L * 1024 * 4);
        Metal.Buffer sc = m.alloc(8L * 1024 * 4);
        // fill: score rows 0..3 = -inf; rows 4..7 = 1,2,3,4 (per row); kv rows 4..7 = row index
        for (int r = 0; r < 8; r++)
            for (int c = 0; c < 1024; c++) {
                kv.setFloat((r * 1024L + c) * 4, r < 4 ? 0f : r);
                sc.setFloat((r * 1024L + c) * 4, r < 4 ? -1.0e30f : r - 3); // rows 4..7 score = 1..4
            }
        try (Metal.Batch bb = m.begin()) {
            tl.launch(bb, tl.POOL_RATIO4_512, 4, 1, 1, tl.args(tl.POOL_RATIO4_512),
                    Tl.B.b(dst, 0), Tl.B.b(kv, 0), Tl.B.b(sc, 0));
            bb.commit();
            bb.waitGpu();
        }
        // expected: softmax(1,2,3,4 over 4 chunks) * rows {4,5,6,7} second halves
        // weights: e^(s-4) / sum -> [e^-3, e^-2, e^-1, 1] / 1.5514
        double z = Math.exp(-3) + Math.exp(-2) + Math.exp(-1) + 1;
        double want = (Math.exp(-3) * 4 + Math.exp(-2) * 5 + Math.exp(-1) * 6 + 1 * 7) / z;
        System.out.printf("dst[0] = %.6f (want %.6f)%n", dst.getFloat(0), want);
        System.out.printf("dst[100] = %.6f (want %.6f)%n", dst.getFloat(100 * 4), want);
    }
}
