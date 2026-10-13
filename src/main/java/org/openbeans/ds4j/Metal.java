package org.openbeans.ds4j;


import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.HashMap;
import java.util.Map;

/**
 * Object-oriented face of the Metal bridge: device, runtime-compiled
 * library, pipeline cache, unified-memory buffers (CPU-mapped so Java
 * reads/writes them zero-copy), and command buffer/encoder helpers.
 */
public final class Metal {
    public final long device;
    public final long queue;
    private final long library;
    private final Map<String, Long> pipelines = new HashMap<>();
    public final String deviceName;

    private Metal(long device, long queue, long library, String name) {
        this.device = device;
        this.queue = queue;
        this.library = library;
        this.deviceName = name;
    }

    /** Open with an explicit MTL language version (0 = runtime default). */
    public static Metal openV(String mslSource, String macros, long langVersion) {
        long dev = Bridge.deviceCreate();
        if (dev == 0) throw new RuntimeException("no Metal device");
        String name = Bridge.deviceName(dev);
        long queue = Bridge.queueCreate(dev);
        long lib = Bridge.libraryCompileV(dev, mslSource, macros, langVersion);
        return new Metal(dev, queue, lib, name);
    }

    /** Creates the system default device and compiles the MSL source at runtime. */
    public static Metal open(String mslSource, String macros) {
        long dev = Bridge.deviceCreate();
        if (dev == 0) throw new RuntimeException("no Metal device");
        String name = Bridge.deviceName(dev);
        long queue = Bridge.queueCreate(dev);
        long lib = Bridge.libraryCompile(dev, mslSource, macros);
        return new Metal(dev, queue, lib, name);
    }

    public long pipeline(String symbol) {
        synchronized (pipelines) {
            return pipelines.computeIfAbsent(symbol, s -> Bridge.pipelineCreate(device, library, s));
        }
    }

    /** A shared-storage Metal buffer whose contents Java maps zero-copy. */
    public static final class Buffer {
        public final long handle;
        public final long bytes;
        /** CPU-visible view of the same storage the GPU reads/writes. */
        public final ByteBuffer bb;

        Buffer(long handle, long bytes, ByteBuffer bb) {
            this.handle = handle;
            this.bytes = bytes;
            this.bb = bb;
        }

        public float getFloat(long byteOffset) { return bb.getFloat((int) byteOffset); }
        public void setFloat(long byteOffset, float v) { bb.putFloat((int) byteOffset, v); }
        public int getInt(long byteOffset) { return bb.getInt((int) byteOffset); }
        public void setInt(long byteOffset, int v) { bb.putInt((int) byteOffset, v); }
        public short getShort(long off) { return bb.getShort((int) off); }
        public void setShort(long off, short v) { bb.putShort((int) off, v); }
        public void setByte(long off, byte v) { bb.put((int) off, v); }

        public float getF16(long byteOffset) {
            return halfToFloat(bb.getShort((int) byteOffset) & 0xffff);
        }

        public void setF16(long byteOffset, float v) {
            bb.putShort((int) byteOffset, (short) floatToHalf(v));
        }

        public static float halfToFloat(int h) {
            int sign = (h & 0x8000) << 16;
            int exp = (h >> 10) & 0x1f;
            int man = h & 0x3ff;
            int bits;
            if (exp == 0) {
                if (man == 0) {
                    bits = sign;
                } else { // subnormal: value = man * 2^-24
                    int s = 0;
                    int m = man;
                    while ((m & 0x400) == 0) { m <<= 1; s++; }
                    // m has bit 10 set: value = 1.frac * 2^(-14-s)
                    bits = sign | ((113 - s) << 23) | ((m & 0x3ff) << 13);
                }
            } else if (exp == 31) {
                bits = sign | 0x7f800000 | (man << 13);
            } else {
                bits = sign | ((exp - 15 + 127) << 23) | (man << 13);
            }
            return Float.intBitsToFloat(bits);
        }

        public static int floatToHalf(float f) {
            int bits = Float.floatToIntBits(f);
            int sign = (bits >> 16) & 0x8000;
            int exp = ((bits >> 23) & 0xff) - 127 + 15;
            int man = bits & 0x7fffff;
            if (((bits >> 23) & 0xff) == 0xff) return sign | 0x7c00; // inf/nan
            if (exp >= 31) return sign | 0x7c00;
            if (exp <= 0) {
                if (exp < -10) return sign;
                man |= 0x800000;
                int shift = 14 - exp; // man_half = M >> shift
                int half = man >> shift;
                int rem = man & ((1 << shift) - 1);
                int halfway = 1 << (shift - 1);
                if (rem > halfway || (rem == halfway && (half & 1) == 1)) half++;
                return sign | half;
            }
            int half = (exp << 10) | (man >> 13);
            int rem = man & 0x1fff;
            if (rem > 0x1000 || (rem == 0x1000 && (half & 1) == 1)) half++;
            return sign | half;
        }
    }

    public Buffer alloc(long bytes) {
        long h = Bridge.bufferAlloc(device, bytes);
        if (h == 0)
            throw new OutOfMemoryError("Metal bufferAlloc(" + bytes + ") failed"
                    + " — GPU wired memory exhausted (check co-resident model servers)");
        long ptr = Bridge.bufferContents(h);
        ByteBuffer bb = java.lang.foreign.MemorySegment.ofAddress(ptr).reinterpret(bytes).asByteBuffer().order(ByteOrder.LITTLE_ENDIAN);
        return new Buffer(h, bytes, bb);
    }

    /** One compute command buffer + encoder; encodes and runs a batch of launches. */
    public final class Batch implements AutoCloseable {
        public final long cb;
        public final long enc;
        private boolean ended;

        Batch() {
            cb = Bridge.commandBufferCreate(queue);
            enc = Bridge.encoderCreate(cb);
        }

        public Batch bind(int index, Buffer buf, long offset) {
            Bridge.setBuffer(enc, buf.handle, offset, index);
            return this;
        }

        public Batch bindBytes(int index, long dataAddr, long len) {
            Bridge.setBytes(enc, dataAddr, len, index);
            return this;
        }

        public Batch launch(String symbol, int gx, int gy, int gz, int tx, int ty, int tz) {
            Bridge.setPipeline(enc, pipeline(symbol));
            Bridge.dispatchThreadgroups(enc, gx, gy, gz, tx, ty, tz);
            return this;
        }

        public void commit() {
            if (!ended) {
                Bridge.encoderEnd(enc);
                ended = true;
            }
            Bridge.commit(cb);
        }

        public void waitGpu() {
            Bridge.waitCompleted(cb);
        }

        public double gpuTimeMs() {
            return Bridge.gpuTimeMs(cb);
        }

        @Override public void close() {
            if (!ended) Bridge.encoderEnd(enc);
            Bridge.release(enc);
            Bridge.release(cb);
        }
    }

    public Batch begin() { return new Batch(); }

    /** Encode + submit + wait in one go. */
    public void run(java.util.function.Consumer<Batch> body) {
        try (Batch b = begin()) {
            body.accept(b);
            b.commit();
            b.waitGpu();
        }
    }
}
