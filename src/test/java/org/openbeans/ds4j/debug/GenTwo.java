package org.openbeans.ds4j.debug;

import org.openbeans.ds4j.*;
import java.nio.file.*;
import java.util.*;

/** Two-turn chat fixtures: writes <prefix>.ids (msg1) and <prefix>_full_ids.txt
 *  (msg1+msg2), prints P and suffix size. Usage: GenTwo <prefix> <sys> <msg1> <msg2> */
public class GenTwo {
    public static void main(String[] a) throws Exception {
        Metal metal = Metal.open(MetalSources.source(),
                "DS4_METAL_HAS_TENSOR=1\nDS4_METAL_HC_STABLE=1\nDS4_METAL_NORM_RSQRT_DISABLE=1");
        Tl tl = new Tl(metal, System.getProperty("ds4j.tunedDir"));
        Gguf gguf = Gguf.open(System.getProperty("ds4j.model", "ds4flash.gguf"));
        Tokenizer tok = new Tokenizer(gguf, 0);
        List<Tokenizer.ChatMsg> m1 = List.of(new Tokenizer.ChatMsg("user", a[2]));
        List<Tokenizer.ChatMsg> m2 = List.of(new Tokenizer.ChatMsg("user", a[2]),
                new Tokenizer.ChatMsg("user", a[3]));
        List<Integer> ids1 = tok.encodeChatHistory(m1, a[1], false);
        List<Integer> ids2 = tok.encodeChatHistory(m2, a[1], false);
        StringBuilder sb = new StringBuilder();
        for (int i : ids1) sb.append(i).append('\n');
        Files.writeString(Path.of(a[0] + "_p.txt"), sb.toString());
        sb = new StringBuilder();
        for (int i : ids2) sb.append(i).append('\n');
        Files.writeString(Path.of(a[0] + "_full.txt"), sb.toString());
        int common = 0;
        while (common < ids1.size() && common < ids2.size()
                && ids1.get(common).equals(ids2.get(common))) common++;
        System.out.println("P=" + ids1.size() + " total=" + ids2.size()
                + " common=" + common + " suffix=" + (ids2.size() - common));
        System.out.println("tail of ids1: " + ids1.subList(Math.max(0, ids1.size()-4), ids1.size()));
        System.out.println("head of suffix: " + ids2.subList(ids1.size(), Math.min(ids2.size(), ids1.size()+4)));
    }
}
