package org.openbeans.ds4j;

import java.util.Arrays;
import java.util.Comparator;

/**
 * Sampling ported from ds4.c sample_top_p_min_p / sample_full_vocab with the
 * exact xorshift64* RNG stream (sample_rng_next / sample_rng_f32) so seeded
 * runs match the C engine's sampled tokens bit for bit.
 */
public final class Sampler {
    private long state;

    public Sampler(long seed) { this.state = seed; }

    private long rngNext() {
        long x = state;
        if (x == 0) x = 0x9e3779b97f4a7c15L;
        x ^= x >>> 12;
        x ^= x << 25;
        x ^= x >>> 27;
        state = x;
        return x * 0x2545f4914f6cdd1dL;
    }

    private float rngF32() {
        long x = rngNext();
        return (float) ((x >>> 40) & 0xffffffL) / 16777216.0f;
    }

    /** C: sample_rng_next — for tests that need raw draws. */
    public long rawNext() { return rngNext(); }

    /** C: sample_rng_f32 — 24-bit draw, for tests. */
    public float drawF32() { return rngF32(); }

    public static int argmax(Metal.Buffer logits, int n) {
        int best = 0;
        float bestV = -Float.MAX_VALUE;
        for (int i = 0; i < n; i++) {
            float v = logits.getFloat(i * 4L);
            if (v > bestV) { bestV = v; best = i; }
        }
        return best;
    }

    /** expf: float-rounded exp (double Math.exp rounded to float). */
    static float expf(float x) {
        return (float) Math.exp(x);
    }

    private static final class Cand {
        int id;
        float logit;
        float prob;
    }

    public int sample(Metal.Buffer logits, int n, float temperature,
                      int topK, float topP, float minP) {
        if (temperature <= 0f) return argmax(logits, n);
        if (topP <= 0f || topP > 1f) topP = 1f;
        if (minP < 0f) minP = 0f;
        if (topK <= 0) return sampleFullVocab(logits, n, temperature, topP, minP);

        // top_k path: insertion-sorted top-K logits, then nucleus over them
        if (topK > 1024) topK = 1024;
        if (topK > n) topK = n;
        int[] ids = new int[topK];
        float[] vals = new float[topK];
        int cnt = 0;
        for (int i = 0; i < n; i++) {
            float v = logits.getFloat(i * 4L);
            if (!Float.isFinite(v)) continue;
            if (cnt == topK && v <= vals[cnt - 1]) continue;
            int j = cnt < topK ? cnt++ : cnt - 1;
            while (j > 0 && vals[j - 1] < v) {
                vals[j] = vals[j - 1];
                ids[j] = ids[j - 1];
                j--;
            }
            vals[j] = v;
            ids[j] = i;
        }
        if (cnt == 0) return argmax(logits, n);

        float[] probs = new float[cnt];
        float sum = 0f;
        for (int i = 0; i < cnt; i++) {
            probs[i] = expf((vals[i] - vals[0]) / temperature);
            sum += probs[i];
        }
        if (sum <= 0f || !Float.isFinite(sum)) return ids[0];

        float minProb = (probs[0] / sum) * minP;
        float filteredSum = 0f;
        int filtered = 0;
        for (int i = 0; i < cnt; i++) {
            float p = probs[i] / sum;
            if (i > 0 && p < minProb) break;
            filteredSum += probs[i];
            filtered++;
            if (filteredSum / sum >= topP) break;
        }
        if (filtered <= 0) return ids[0];

        float r = rngF32() * filteredSum;
        for (int i = 0; i < filtered; i++) {
            r -= probs[i];
            if (r <= 0f) return ids[i];
        }
        return ids[filtered - 1];
    }

    private int sampleFullVocab(Metal.Buffer logits, int n, float temperature,
                                float topP, float minP) {
        float maxLogit = -Float.MAX_VALUE;
        int best = 0;
        int finite = 0;
        for (int i = 0; i < n; i++) {
            float v = logits.getFloat(i * 4L);
            if (!Float.isFinite(v)) continue;
            finite++;
            if (v > maxLogit) { maxLogit = v; best = i; }
        }
        if (finite == 0) return argmax(logits, n);

        if (topP >= 1.0f) {
            // min-p as an absolute (unnormalized) floor on exp((v-max)/T),
            // one RNG draw, walk in id order over the survivors
            float minRel = minP > 0f ? minP : 0f;
            if (minRel > 1f) return best;
            float[] p = new float[n];
            float sum = 0f;
            for (int i = 0; i < n; i++) {
                p[i] = -1f;
                float v = logits.getFloat(i * 4L);
                if (!Float.isFinite(v)) continue;
                float e = expf((v - maxLogit) / temperature);
                if (e < minRel) continue;
                p[i] = e;
                sum += e;
            }
            if (sum <= 0f || !Float.isFinite(sum)) return best;
            float r = rngF32() * sum;
            for (int i = 0; i < n; i++) {
                float e = p[i];
                if (e < 0f) continue;
                r -= e;
                if (r <= 0f) return i;
            }
            return best;
        }

        // top_p < 1: sort candidates by logit desc / id asc, nucleus cut
        float[] p = new float[n];
        float sum = 0f;
        for (int i = 0; i < n; i++) {
            p[i] = -1f;
            float v = logits.getFloat(i * 4L);
            if (!Float.isFinite(v)) continue;
            float e = expf((v - maxLogit) / temperature);
            p[i] = e;
            sum += e;
        }
        if (sum <= 0f || !Float.isFinite(sum)) return best;

        Cand[] cand = new Cand[finite];
        int cnt = 0;
        if (minP > 0f && minP <= 1f) {
            float minProb = (1f / sum) * minP;
            for (int i = 0; i < n; i++) {
                float e = p[i];
                if (e < 0f || e / sum < minProb) continue;
                Cand c = new Cand();
                c.id = i;
                c.logit = logits.getFloat(i * 4L);
                c.prob = e;
                cand[cnt++] = c;
            }
            if (cnt == 0) return best;
        } else {
            for (int i = 0; i < n; i++) {
                float e = p[i];
                if (e < 0f) continue;
                Cand c = new Cand();
                c.id = i;
                c.logit = logits.getFloat(i * 4L);
                c.prob = e;
                cand[cnt++] = c;
            }
        }
        Arrays.sort(cand, 0, cnt, Comparator
                .<Cand>comparingDouble(c -> -(double) c.logit)
                .thenComparingInt(c -> c.id));

        float minProb = (cand[0].prob / sum) * minP;
        float filteredSum = 0f;
        int filtered = 0;
        for (int i = 0; i < cnt; i++) {
            float pv = cand[i].prob / sum;
            if (i > 0 && pv < minProb) break;
            filteredSum += cand[i].prob;
            filtered++;
            if (filteredSum / sum >= topP) break;
        }
        if (filtered == 0) return best;

        float r = rngF32() * filteredSum;
        for (int i = 0; i < filtered; i++) {
            r -= cand[i].prob;
            if (r <= 0f) return cand[i].id;
        }
        return cand[filtered - 1].id;
    }
}
