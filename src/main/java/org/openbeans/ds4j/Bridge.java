package org.openbeans.ds4j;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.GroupLayout;
import java.lang.foreign.Linker;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.HashMap;

/**
 * The native bridge: the JDK's Foreign Function & Memory API downcalls
 * straight into the Objective-C runtime (libobjc) and the Metal framework.
 * No native shim, no third-party libraries — the JDK is the only runtime
 * dependency. objc_msgSend is one symbol; each call
 * shape used by the engine gets its own pre-linked MethodHandle stub, so the
 * hot path (setPipeline/setBuffer/setBytes/dispatch) pays only the FFM
 * downcall cost (~tens of ns per call — the same class of cost the C shim
 * path pays, minus the C intermediary). MTLSize structs pass by value via
 * GroupLayout descriptors (arm64 passes 3x-integer trivial structs in
 * registers; Linker.Option.isTrivial marks them as such). Model weights
 * map through a plain FileChannel.
 *
 * Lifetime discipline: class-factory results are
 * autoreleased +0 references; with no autorelease pool they can be zombied
 * by pool drains that happen inside Metal, so every object we keep is
 * explicitly retained, and release() really releases.
 */
public final class Bridge {

    private static final Linker L = Linker.nativeLinker();
    private static final Arena ARENA = Arena.ofShared();
    private static final SymbolLookup OBJC = SymbolLookup.libraryLookup("/usr/lib/libobjc.dylib", ARENA);
    private static final SymbolLookup METAL = SymbolLookup.libraryLookup(
            "/System/Library/Frameworks/Metal.framework/Metal", ARENA);

    private static final ValueLayout A = ValueLayout.ADDRESS;
    private static final ValueLayout.OfLong J = ValueLayout.JAVA_LONG;

    private static final MemorySegment MSG = OBJC.find("objc_msgSend").orElseThrow();
    private static final MemorySegment GET_CLASS = OBJC.find("objc_getClass").orElseThrow();
    private static final MemorySegment SEL_REG = OBJC.find("sel_registerName").orElseThrow();
    private static final MemorySegment MTL_DEVICE =
            METAL.find("MTLCreateSystemDefaultDevice").orElseThrow();

    /** MTLSize {NSUInteger width, height, depth} — passed by value. */
    private static final GroupLayout MTL_SIZE =
            MemoryLayout.structLayout(J.withName("w"), J.withName("h"), J.withName("d"));

    // ---- objc_msgSend stubs, one per call shape ----
    private static final MethodHandle hId2   = down(FunctionDescriptor.of(A, A, A));
    private static final MethodHandle hV2   = down(FunctionDescriptor.ofVoid(A, A));
    private static final MethodHandle hD2   = down(FunctionDescriptor.of(ValueLayout.JAVA_DOUBLE, A, A));
    private static final MethodHandle hL2   = down(FunctionDescriptor.of(J, A, A));
    private static final MethodHandle hI3A  = down(FunctionDescriptor.of(ValueLayout.JAVA_INT, A, A, A));
    private static final MethodHandle hId3A = down(FunctionDescriptor.of(A, A, A, A));
    private static final MethodHandle hV3A  = down(FunctionDescriptor.ofVoid(A, A, A));
    private static final MethodHandle hId3J = down(FunctionDescriptor.of(A, A, A, J));
    private static final MethodHandle hV3J  = down(FunctionDescriptor.ofVoid(A, A, J));
    private static final MethodHandle hId4JJ = down(FunctionDescriptor.of(A, A, A, J, J));
    private static final MethodHandle hV4AJJ = down(FunctionDescriptor.ofVoid(A, A, A, J, J));
    private static final MethodHandle hV4AA  = down(FunctionDescriptor.ofVoid(A, A, A, A));
    private static final MethodHandle hId4AA = down(FunctionDescriptor.of(A, A, A, A, A));
    private static final MethodHandle hId5AAA = down(FunctionDescriptor.of(A, A, A, A, A, A));
    private static final MethodHandle hId4PAJJA =
            down(FunctionDescriptor.of(A, A, A, A, J, J, A));
    // (no Linker.Option needed on JDK 25: the linker classifies the 3x-long
    // struct as INTEGER registers per the arm64 ABI on its own)
    private static final MethodHandle hV4SS =
            down(FunctionDescriptor.ofVoid(A, A, MTL_SIZE, MTL_SIZE));

    private static MethodHandle down(FunctionDescriptor d, Linker.Option... opts) {
        return L.downcallHandle(MSG, d, opts);
    }

    // ---- cached classes / selectors (stable MemorySegments) ----
    private static final HashMap<String, MemorySegment> CLS = new HashMap<>();
    private static final HashMap<String, MemorySegment> SEL = new HashMap<>();

    private static MemorySegment cls(String n) {
        return CLS.computeIfAbsent(n, k -> {
            try (Arena a = Arena.ofConfined()) {
                MemorySegment z = a.allocate(k.getBytes(StandardCharsets.UTF_8).length + 1);
                z.copyFrom(MemorySegment.ofArray(k.getBytes(StandardCharsets.UTF_8)));
                return (MemorySegment) L.downcallHandle(GET_CLASS,
                        FunctionDescriptor.of(A, A)).invokeExact(z);
            } catch (Throwable t) { throw sneaky(t); }
        });
    }

    private static MemorySegment sel(String n) {
        return SEL.computeIfAbsent(n, k -> {
            try (Arena a = Arena.ofConfined()) {
                MemorySegment z = a.allocate(k.getBytes(StandardCharsets.UTF_8).length + 1);
                z.copyFrom(MemorySegment.ofArray(k.getBytes(StandardCharsets.UTF_8)));
                return (MemorySegment) L.downcallHandle(SEL_REG,
                        FunctionDescriptor.of(A, A)).invokeExact(z);
            } catch (Throwable t) { throw sneaky(t); }
        });
    }

    private static final MemorySegment
            S_RETAIN = sel("retain"),
            S_RELEASE = sel("release"),
            S_SET_PIPELINE = sel("setComputePipelineState:"),
            S_SET_BUFFER = sel("setBuffer:offset:atIndex:"),
            S_SET_BYTES = sel("setBytes:length:atIndex:"),
            S_DISPATCH = sel("dispatchThreadgroups:threadsPerThreadgroup:"),
            S_END = sel("endEncoding"),
            S_COMMIT = sel("commit"),
            S_WAIT = sel("waitUntilCompleted"),
            S_GPU_END = sel("GPUEndTime"),
            S_GPU_START = sel("GPUStartTime");

    static RuntimeException sneaky(Throwable t) {
        if (t instanceof RuntimeException r) return r;
        if (t instanceof Error e) throw e;
        return new RuntimeException(t);
    }

    private static MemorySegment of(long a) { return MemorySegment.ofAddress(a); }

    private static long send2(long a, MemorySegment s) {
        try { return ((MemorySegment) hId2.invokeExact(of(a), s)).address(); }
        catch (Throwable t) { throw sneaky(t); }
    }
    private static void void2(long a, MemorySegment s) {
        try { hV2.invokeExact(of(a), s); }
        catch (Throwable t) { throw sneaky(t); }
    }

    /** Class-factory object we intend to keep: retain to escape autorelease. */
    static long retain(long obj) {
        try { return ((MemorySegment) hId2.invokeExact(of(obj), S_RETAIN)).address(); }
        catch (Throwable t) { throw sneaky(t); }
    }

    static long nsstring(String s) {
        byte[] b = s.getBytes(StandardCharsets.UTF_8);
        MemorySegment z = ARENA.allocate(b.length + 1);
        z.copyFrom(MemorySegment.ofArray(b));
        try {
            MemorySegment r = (MemorySegment) hId3A.invokeExact(
                    cls("NSString"), sel("stringWithUTF8String:"), z);
            return retain(r.address());
        } catch (Throwable t) { throw sneaky(t); }
    }

    private static String utf8String(long obj) {
        try {
            long c = ((MemorySegment) hId2.invokeExact(of(obj), sel("UTF8String"))).address();
            return c == 0 ? "" : MemorySegment.ofAddress(c).reinterpret(Long.MAX_VALUE).getString(0);
        } catch (Throwable t) { throw sneaky(t); }
    }

    // ---- device / queue ----

    public static long deviceCreate() {
        try {
            return ((MemorySegment) L.downcallHandle(MTL_DEVICE, FunctionDescriptor.of(A)).invokeExact())
                    .address();
        } catch (Throwable t) { throw sneaky(t); }
    }

    public static String deviceName(long dev) {
        return utf8String(send2(dev, sel("name")));
    }

    public static long queueCreate(long dev) {
        try {
            MemorySegment q = (MemorySegment) hId3J.invokeExact(
                    of(dev), sel("newCommandQueueWithMaxCommandBufferCount:"), 256L);
            return q.address(); // +1 factory
        } catch (Throwable t) { throw sneaky(t); }
    }

    // ---- runtime shader compilation ----

    public static long libraryCompile(long dev, String source, String macros) {
        return libraryCompileImpl(dev, source, macros, 0x40000L);
    }

    public static long libraryCompileV(long dev, String source, String macros, long langVersion) {
        return libraryCompileImpl(dev, source, macros, langVersion);
    }

    static long libraryCompileImpl(long dev, String source, String macros, long langVersion) {
        long srcStr = nsstring(source);

        long opts = send2(cls("MTLCompileOptions").address(), sel("alloc"));
        opts = send2(opts, sel("init"));
        try { hV3J.invokeExact(of(opts), sel("setLanguageVersion:"), langVersion); }
        catch (Throwable t) { throw sneaky(t); }

        if (macros != null && !macros.isEmpty()) {
            long dict = send2(cls("NSMutableDictionary").address(), sel("alloc"));
            dict = send2(dict, sel("init"));
            for (String line : macros.split("\n")) {
                int eq = line.indexOf('=');
                if (eq < 0) continue;
                long k = nsstring(line.substring(0, eq));
                long v = nsstring(line.substring(eq + 1));
                try { hV4AA.invokeExact(of(dict), sel("setObject:forKey:"), of(v), of(k)); }
                catch (Throwable t) { throw sneaky(t); }
            }
            try { hV3A.invokeExact(of(opts), sel("setPreprocessorMacros:"), of(dict)); }
            catch (Throwable t) { throw sneaky(t); }
            void2(dict, S_RELEASE);
        }

        MemorySegment err = ARENA.allocate(8);
        long lib;
        try {
            lib = ((MemorySegment) hId5AAA.invokeExact(of(dev),
                    sel("newLibraryWithSource:options:error:"), of(srcStr), of(opts), err)).address();
        } catch (Throwable t) { throw sneaky(t); }
        void2(opts, S_RELEASE);
        void2(srcStr, S_RELEASE);
        if (lib == 0) {
            long e = err.getAtIndex(J, 0);
            String msg = e != 0 ? utf8String(send2(e, sel("localizedDescription"))) : "(unknown)";
            throw new RuntimeException("Metal library compile failed: " + msg);
        }
        return lib;
    }

    public static long pipelineCreate(long dev, long lib, String name) {
        long nameStr = nsstring(name);
        long fn;
        try {
            fn = ((MemorySegment) hId3A.invokeExact(of(lib), sel("newFunctionWithName:"), of(nameStr))).address();
        } catch (Throwable t) { throw sneaky(t); }
        void2(nameStr, S_RELEASE);
        if (fn == 0) throw new RuntimeException("function not found: " + name);
        MemorySegment err = ARENA.allocate(8);
        long pso;
        try {
            pso = ((MemorySegment) hId4AA.invokeExact(of(dev),
                    sel("newComputePipelineStateWithFunction:error:"), of(fn), err)).address();
        } catch (Throwable t) { throw sneaky(t); }
        if (pso == 0) {
            long e = err.getAtIndex(J, 0);
            String msg = e != 0 ? utf8String(send2(e, sel("localizedDescription"))) : "(unknown)";
            throw new RuntimeException("pipeline creation failed: " + msg);
        }
        return pso;
    }

    // ---- buffers ----

    public static long bufferAlloc(long dev, long bytes) {
        try {
            MemorySegment r = (MemorySegment) hId4JJ.invokeExact(
                    of(dev), sel("newBufferWithLength:options:"), bytes, 0L);
            return r.address(); // +1 factory
        } catch (Throwable t) { throw sneaky(t); }
    }

    public static long bufferWrap(long dev, long addr, long bytes) {
        try {
            MemorySegment r = (MemorySegment) hId4PAJJA.invokeExact(of(dev),
                    sel("newBufferWithBytesNoCopy:length:options:deallocator:"),
                    of(addr), bytes, 0L, MemorySegment.NULL);
            return r.address(); // +1 factory
        } catch (Throwable t) { throw sneaky(t); }
    }

    public static long bufferContents(long buf) {
        return send2(buf, sel("contents"));
    }

    public static long deviceMaxBufferLength(long dev) {
        try { return (long) hL2.invokeExact(of(dev), sel("maxBufferLength")); }
        catch (Throwable t) { throw sneaky(t); }
    }

    /** Pure-Java mmap of one view of the weight file; page-aligned position;
     *  returned address kept alive via the MAPPINGS references. */
    public static long mmapView(String path, long fileOffset, long bytes) {
        try (FileChannel ch = FileChannel.open(Path.of(path), StandardOpenOption.READ)) {
            long cap = Math.min(bytes, ch.size() - fileOffset);
            java.nio.ByteBuffer map = ch.map(FileChannel.MapMode.READ_ONLY, fileOffset, cap);
            MAPPINGS.add(map);
            return MemorySegment.ofBuffer(map).address();
        } catch (Exception e) {
            throw new RuntimeException("cannot mmap model view at " + fileOffset, e);
        }
    }

    private static final java.util.List<java.nio.ByteBuffer> MAPPINGS = new java.util.ArrayList<>();

    // ---- command recording ----

    public static long commandBufferCreate(long queue) {
        return retain(send2(queue, sel("commandBuffer")));
    }

    public static long encoderCreate(long cb) {
        return retain(send2(cb, sel("computeCommandEncoder")));
    }

    public static void setPipeline(long enc, long pso) {
        try { hV3A.invokeExact(of(enc), S_SET_PIPELINE, of(pso)); }
        catch (Throwable t) { throw sneaky(t); }
    }

    public static void setBuffer(long enc, long buf, long offset, int index) {
        try { hV4AJJ.invokeExact(of(enc), S_SET_BUFFER, of(buf), offset, (long) index); }
        catch (Throwable t) { throw sneaky(t); }
    }

    public static void setBytes(long enc, long dataAddr, long len, int index) {
        try { hV4AJJ.invokeExact(of(enc), S_SET_BYTES, of(dataAddr), len, (long) index); }
        catch (Throwable t) { throw sneaky(t); }
    }

    private static final HashMap<Long, MemorySegment> SIZES = new HashMap<>();

    private static MemorySegment size(int a, int b, int c) {
        long key = ((long) a << 42) | ((long) b << 21) | (c & 0x1FFFFFL);
        return SIZES.computeIfAbsent(key, k -> {
            MemorySegment s = ARENA.allocate(MTL_SIZE);
            s.set(J, 0, a);
            s.set(J, 8, b);
            s.set(J, 16, c);
            return s;
        });
    }

    public static void dispatchThreadgroups(long enc, int gx, int gy, int gz, int tx, int ty, int tz) {
        try { hV4SS.invokeExact(of(enc), S_DISPATCH, size(gx, gy, gz), size(tx, ty, tz)); }
        catch (Throwable t) { throw sneaky(t); }
    }

    public static void encoderEnd(long enc) { void2(enc, S_END); }
    public static void commit(long cb) { void2(cb, S_COMMIT); }
    public static void waitCompleted(long cb) { void2(cb, S_WAIT); }

    public static double gpuTimeMs(long cb) {
        try {
            double end = (double) hD2.invokeExact(of(cb), S_GPU_END);
            double start = (double) hD2.invokeExact(of(cb), S_GPU_START);
            return (end - start) * 1000.0;
        } catch (Throwable t) { throw sneaky(t); }
    }

    public static void release(long obj) {
        if (obj != 0) void2(obj, S_RELEASE);
    }

    // ---- model residency (C parity: MTLResidencySet; macOS 15+) ----

    public static long residencySetCreate(long dev, int capacity) {
        long descCls = cls("MTLResidencySetDescriptor").address();
        if (descCls == 0) return 0;
        long d = send2(descCls, sel("alloc"));
        d = send2(d, sel("init"));
        long label = nsstring("ds4j_model");
        try { hV3A.invokeExact(of(d), sel("setLabel:"), of(label)); }
        catch (Throwable t) { throw sneaky(t); }
        try { hV3J.invokeExact(of(d), sel("setInitialCapacity:"), (long) capacity); }
        catch (Throwable t) { throw sneaky(t); }
        MemorySegment err = ARENA.allocate(8);
        long set;
        try {
            set = ((MemorySegment) hId4AA.invokeExact(of(dev),
                    sel("newResidencySetWithDescriptor:error:"), of(d), err)).address();
        } catch (Throwable t) { throw sneaky(t); }
        void2(label, S_RELEASE);
        void2(d, S_RELEASE);
        return set; // +1 factory
    }

    public static void residencyAdd(long set, long buf) {
        try { hV3A.invokeExact(of(set), sel("addAllocation:"), of(buf)); }
        catch (Throwable t) { throw sneaky(t); }
    }

    public static void residencyCommit(long set) {
        void2(set, sel("commit"));
        void2(set, sel("requestResidency"));
    }

    public static int residencyQueueAdd(long queue, long set) {
        try {
            int ok = (int) hI3A.invokeExact(of(queue),
                    sel("respondsToSelector:"), sel("addResidencySet:"));
            if ((ok & 0xFF) == 0) return 0;
            hV3A.invokeExact(of(queue), sel("addResidencySet:"), of(set));
            return 1;
        } catch (Throwable t) { throw sneaky(t); }
    }

    private Bridge() {}
}