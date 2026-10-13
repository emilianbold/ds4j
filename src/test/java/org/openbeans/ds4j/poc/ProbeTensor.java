package org.openbeans.ds4j.poc;

import org.openbeans.ds4j.Metal;

/** Compile-probe: does the runtime shader compiler accept the Metal 4 tensor API? */
public final class ProbeTensor {
    static final String SRC = """
        #include <metal_stdlib>
        #include <metal_tensor>
        #include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
        using namespace metal;
        using namespace mpp::tensor_ops;
        kernel void ds4_tensor_probe(
                tensor<device half,  dextents<int32_t, 2>> A [[buffer(0)]],
                tensor<device half,  dextents<int32_t, 2>> B [[buffer(1)]],
                device float *C [[buffer(2)]],
                uint2 tgid [[threadgroup_position_in_grid]]) {
            auto tA = A.slice(0, (int)tgid.y);
            auto tB = B.slice((int)tgid.x, 0);
            matmul2d<matmul2d_descriptor(16, 16, dynamic_extent), execution_simdgroups<4>> mm;
            auto cT = mm.get_destination_cooperative_tensor<decltype(tA), decltype(tB), float>();
            auto sA = tA.slice(0, 0);
            auto sB = tB.slice(0, 0);
            mm.run(sB, sA, cT);
            auto tC = tensor<device float, dextents<int32_t, 2>, tensor_inline>(C, dextents<int32_t, 2>(16, 16));
            cT.store(tC);
        }
        """;

    public static void main(String[] a) {
        try {
            Metal m = Metal.open(SRC, null);
            System.out.println("probe: compiled OK, device " + m.deviceName);
        } catch (Throwable t) {
            System.out.println("probe FAILED: " + t.getMessage());
        }
    }
}
