package org.openbeans.ds4j.debug;

import org.openbeans.ds4j.*;
import java.nio.file.*;
import java.util.*;
public class RenderIds {
    public static void main(String[] a) throws Exception {
        Metal metal = Metal.open(MetalSources.source(),
                "DS4_METAL_HAS_TENSOR=1\nDS4_METAL_HC_STABLE=1\nDS4_METAL_NORM_RSQRT_DISABLE=1");
        Tl tl = new Tl(metal, System.getProperty("ds4j.tunedDir"));
        Gguf gguf = Gguf.open(System.getProperty("ds4j.model", "ds4flash.gguf"));
        Tokenizer tok = new Tokenizer(gguf, 0);
        String text = Files.readString(Path.of(a[0]));
        List<Integer> ids = tok.encodeChatText(text);
        StringBuilder sb = new StringBuilder();
        for (int i : ids) sb.append(i).append('\n');
        Files.writeString(Path.of(a[1]), sb.toString());
        System.out.println("rendered " + ids.size() + " tokens");
    }
}
