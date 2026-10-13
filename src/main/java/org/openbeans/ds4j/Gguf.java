package org.openbeans.ds4j;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Minimal GGUF v3 reader: parses the metadata KV table and the tensor
 * directory from the file header region (pure java.nio, no memory mapping of
 * the metadata). Weight bytes are never copied; tensors are (offset, length)
 * windows into the file, later exposed to Metal as no-copy views of a plain
 * Java mmap (see ObjcBridge.mmapData).
 */
public final class Gguf {
    public static final int T_F32 = 0, T_F16 = 1, T_Q8_0 = 8, T_Q2_K = 10,
            T_IQ2_XXS = 16, T_I32 = 26, T_BF16 = 30;

    public record Tensor(String name, int ndim, long[] dims, int type,
                         long relOffset, long absOffset, long elements, long bytes) {}

    public final long size;
    public final String path;
    public final Map<String, Object> meta = new HashMap<>();
    public final Map<String, Tensor> tensors = new HashMap<>();
    public final long dataStart;
    /** little-endian view of the parsed header region (diagnostics). */
    public final ByteBuffer head;

    private Gguf(String pathIn) {
        path = pathIn;
        try (FileChannel ch = FileChannel.open(Path.of(path), StandardOpenOption.READ)) {
            size = ch.size();
            long cap = Math.min(size, 64L << 20);
            head = ByteBuffer.allocate((int) cap).order(ByteOrder.LITTLE_ENDIAN);
            long got = 0;
            while (got < cap) {
                int r = ch.read(head, got);
                if (r < 0) break;
                got += r;
            }
        } catch (Exception e) {
            throw new RuntimeException("cannot open GGUF " + path, e);
        }

        int magic = head.getInt(0);
        if (magic != 0x46554747) throw new RuntimeException("not a GGUF file");
        long version = u32(4);
        if (version != 3) throw new RuntimeException("unsupported GGUF version " + version);
        long nTensors = u64(8);
        long nKv = u64(16);
        pos = 24;
        for (long i = 0; i < nKv; i++) readKv();
        for (long i = 0; i < nTensors; i++) readTensor();
        long alignment = optLong("general.alignment", 32);
        dataStart = alignUp(pos, alignment);
        patch();
    }

    private static long alignUp(long v, long a) { return (v + a - 1) / a * a; }

    private long pos;

    private long u8() { return head.get((int) pos++) & 0xffL; }
    private long u32() { long v = u8() | u8() << 8 | u8() << 16 | u8() << 24; return v; }
    private long u32(long at) { return Integer.toUnsignedLong(head.getInt((int) at)); }
    private long u64() { return u32() | u32() << 32; }
    private long u64(long at) { return head.getLong((int) at); }

    private String str() {
        long len = u64();
        if (len == 0) return "";
        byte[] b = new byte[(int) len];
        for (int i = 0; i < len; i++) b[i] = (byte) u8();
        return new String(b, java.nio.charset.StandardCharsets.UTF_8);
    }

    private static final int M_UINT8 = 0, M_INT8 = 1, M_UINT16 = 2, M_INT16 = 3,
            M_UINT32 = 4, M_INT32 = 5, M_FLOAT32 = 6, M_BOOL = 7,
            M_STRING = 8, M_ARRAY = 9, M_UINT64 = 10, M_INT64 = 11, M_FLOAT64 = 12;

    private Object scalar(int t) {
        return switch (t) {
            case M_UINT8 -> (short) u8();
            case M_INT8 -> (byte) u8();
            case M_UINT16 -> (int) (u8() | u8() << 8);
            case M_INT16 -> (short) (u8() | u8() << 8);
            case M_UINT32 -> (long) u32();
            case M_INT32 -> (int) u32();
            case M_FLOAT32 -> Float.intBitsToFloat((int) u32());
            case M_BOOL -> u8() != 0;
            case M_STRING -> str();
            case M_UINT64 -> u64();
            case M_INT64 -> u64();
            case M_FLOAT64 -> Double.longBitsToDouble(u64());
            default -> throw new RuntimeException("bad scalar type " + t);
        };
    }

    private void readKv() {
        String key = str();
        int type = (int) u32();
        Object value = switch (type) {
            case M_ARRAY -> {
                int et = (int) u32();
                long n = u64();
                Object[] a = new Object[(int) n];
                for (int i = 0; i < n; i++) a[i] = scalar(et);
                yield a;
            }
            default -> scalar(type);
        };
        meta.put(key, value);
    }

    private final List<Tensor> pendingAbs = new ArrayList<>();

    private void readTensor() {
        String name = str();
        int ndim = (int) u32();
        long[] dims = new long[ndim];
        long elements = 1;
        for (int i = 0; i < ndim; i++) {
            dims[i] = u64();
            elements *= dims[i];
        }
        int type = (int) u32();
        long rel = u64();
        long bytes = typeBytes(type, elements);
        Tensor t = new Tensor(name, ndim, dims, type, rel, -1, elements, bytes);
        tensors.put(name, t);
        pendingAbs.add(t);
    }

    public static long typeBytes(int type, long elements) {
        return switch (type) {
            case T_F32, T_I32 -> 4 * elements;
            case T_F16, T_BF16 -> 2 * elements;
            case T_Q8_0 -> elements / 32 * 34;
            case T_Q2_K -> elements / 256 * 84;
            case T_IQ2_XXS -> elements / 256 * 66;
            default -> -1;
        };
    }

    public long optLong(String key, long def) {
        Object v = meta.get(key);
        if (v instanceof Number n) return n.longValue();
        return def;
    }
    public String optStr(String key, String def) {
        Object v = meta.get(key);
        return v instanceof String s ? s : def;
    }
    public boolean has(String key) { return meta.containsKey(key); }

    public Tensor t(String name) {
        Tensor t = tensors.get(name);
        if (t == null || t.absOffset() < 0)
            throw new RuntimeException("missing tensor: " + name);
        return t;
    }

    public Tensor find(String name) {
        Tensor t = tensors.get(name);
        return t == null || t.absOffset() < 0 ? null : t;
    }

    private void patch() {
        for (Tensor t : pendingAbs) {
            Tensor t2 = new Tensor(t.name(), t.ndim(), t.dims(), t.type(), t.relOffset(),
                    dataStart + t.relOffset(), t.elements(), t.bytes());
            tensors.put(t.name(), t2);
        }
    }

    public static Gguf open(String path) {
        return new Gguf(path);
    }
}
