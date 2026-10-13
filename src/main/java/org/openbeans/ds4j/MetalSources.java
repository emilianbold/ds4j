package org.openbeans.ds4j;

import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Runtime resource loading for the packaged engine: the TileLang Metal
 * kernel sources, the per-GPU autotune selection JSON, and the IQ2_XXS
 * codebook header. Each item is read from the classpath first (the jar
 * packs them under metal/, tuned/, iq2/ — see java/Makefile `resources`)
 * and falls back to the repository layout for development runs from
 * java/build, so the same binaries work unpackaged and packaged.
 */
public final class MetalSources {

    static final String[] METAL_FILES = {
            "tl_prelude", "tl_norm", "tl_q8", "tl_moe", "tl_dsv4",
            "tl_elementwise", "tl_utility", "tl_router", "tl_compressor",
            "tl_attention", "tl_indexer", "tl_q8gemm",
    };

    /** Concatenated Metal source: prelude then gen/metal/*.metal in the
     *  fixed order (later files reference earlier definitions). */
    public static String source() {
        StringBuilder sb = new StringBuilder(1 << 21);
        sb.append(read("metal/prelude.metal")).append('\n');
        for (String f : METAL_FILES)
            sb.append(read("metal/gen/" + f + ".metal")).append('\n');
        return sb.toString();
    }

    /** Autotune selection JSON for this GPU (e.g. tuned/apple-m5-max.json). */
    public static String tunedJson(String gpuKey) {
        String name = "tuned/" + gpuKey + ".json";
        return readResource(name);
    }

    /** IQ2_XXS codebook header (static dequant tables; parsed once). */
    public static String iq2Tables() {
        String explicit = System.getProperty("ds4j.iq2tables");
        if (explicit != null) {
            try { return Files.readString(Path.of(explicit)); } catch (Exception e) {
                throw new RuntimeException("cannot read " + explicit, e);
            }
        }
        String fromCp = readResource("iq2/ds4_iq2_tables.h");
        if (fromCp == null)
            throw new RuntimeException("ds4_iq2_tables.h not found on the classpath");
        return fromCp;
    }

    /** Classpath-only read: resources are packaged in the jar, or live in
     * target/classes after `mvn package` / `mvn test-compile`. */
    static String read(String resourcePath) {
        String fromCp = readResource(resourcePath);
        if (fromCp == null)
            throw new IllegalStateException("missing resource " + resourcePath
                    + " (not on the classpath; use the packaged jar or target/classes)");
        return fromCp;
    }

    static String readResource(String path) {
        try (InputStream in = MetalSources.class.getResourceAsStream("/" + path)) {
            return in == null ? null : new String(in.readAllBytes(), StandardCharsets.UTF_8);
        } catch (Exception e) { return null; }
    }

    private MetalSources() {}
}