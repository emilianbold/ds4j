package org.openbeans.ds4j.debug;

import org.openbeans.ds4j.*;
/** Per-token wall vs GPU time to show host-side overhead. */
public final class Bench {
    public static void main(String[] a) throws Exception {
        Metal metal = Metal.open(MetalSources.source(),
                "DS4_METAL_HAS_TENSOR=1\nDS4_METAL_HC_STABLE=1\nDS4_METAL_NORM_RSQRT_DISABLE=1");
        Tl tl = new Tl(metal, System.getProperty("ds4j.tunedDir"));
        Gguf gguf = Gguf.open(System.getProperty("ds4j.model", "ds4flash.gguf"));
        Engine e = new Engine(metal, tl, gguf, 1024);
        Tokenizer tok = new Tokenizer(gguf, e.nVocab);
        var ids = tok.encodeChat("You are a helpful assistant", "Count from one to twenty.", false);
        int tokId = 0;
        double wallSum = 0, gpuSum = 0, encSum = 0;
        int n = 0;
        for (int pos = 0; pos < ids.size() + 60; pos++) {
            long t0 = System.nanoTime();
            Metal.Batch bb = metal.begin();
            long t1 = System.nanoTime();
            e.decode(bb, pos < ids.size() ? ids.get(pos) : tokId, pos, pos >= ids.size() - 1);
            long t2 = System.nanoTime(); // decode() commits+waits inside
            long t3 = System.nanoTime();
            double gpu = bb.gpuTimeMs();
            if (pos >= ids.size()) {
                wallSum += (t3 - t0) / 1e6; gpuSum += gpu; encSum += ((t2 - t1) / 1e6);
                n++;
                tokId = Sampler.argmax(e.logits, e.nVocab);
            }
            bb.close();
        }
        System.out.printf("tokens=%d  wall %.2f ms/tok  gpu %.2f ms/tok  encode %.2f ms/tok  => host overhead %.2f ms/tok%n",
                n, wallSum / n, gpuSum / n, encSum / n, (wallSum - gpuSum) / n);
    }
}
