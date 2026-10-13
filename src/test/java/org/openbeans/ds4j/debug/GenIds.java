package org.openbeans.ds4j.debug;

import org.openbeans.ds4j.*;
import java.nio.file.*;
import java.util.*;

public class GenIds {
    public static void main(String[] a) throws Exception {
        String model = System.getProperty("ds4j.model", "ds4flash.gguf");
        String src = MetalSources.source();
        Metal metal = Metal.open(src,
                "DS4_METAL_HAS_TENSOR=1\nDS4_METAL_HC_STABLE=1\nDS4_METAL_NORM_RSQRT_DISABLE=1");
        Tl tl = new Tl(metal, System.getProperty("ds4j.tunedDir"));
        Gguf gguf = Gguf.open(model);
        Tokenizer tok = new Tokenizer(gguf, 0);
        // args: <out-file> <system> <user> [think]
        String system = a.length > 1 ? a[1] : "You are a helpful assistant";
        String user = a.length > 2 ? a[2] : "Say hello and tell me what model you are.";
        boolean think = a.length > 3 && a[3].equals("think");
        List<Integer> ids = tok.encodeChat(system, user, think);
        StringBuilder sb = new StringBuilder();
        for (int i : ids) sb.append(i).append('\n');
        Files.writeString(Path.of(a[0]), sb.toString());
        System.out.println("n=" + ids.size() + " ids: " + ids);
    }
}
