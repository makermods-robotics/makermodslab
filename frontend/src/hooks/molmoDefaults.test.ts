import { act, renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";
import { useGpuKnobs } from "./useGpuLauncher";
import { remoteDefaultsForPolicy } from "@/components/remote-inference/remoteRunConfig";

beforeEach(() => localStorage.clear());

describe("MolmoAct2 defaults", () => {
  it("uses the tested timing and filter without changing other policies", () => {
    expect(remoteDefaultsForPolicy({ policy_type: "molmoact2" })).toEqual({ fps: 20, videoCodec: "MJPEG", sMin: 4, lpfHz: 2 });
    expect(remoteDefaultsForPolicy({ policy_type: "act" })).toEqual({ fps: 30, videoCodec: "H264", sMin: 4, lpfHz: 0 });
  });

  it("keeps older GPU picks separate and restores explicit Molmo overrides", () => {
    localStorage.setItem("makermodslab.gpuType", "A100");
    localStorage.setItem("makermodslab.gpuModelDtype", "float32");
    const { result, rerender, unmount } = renderHook(({ policy }) => useGpuKnobs(policy), { initialProps: { policy: "molmoact2" } });
    expect(result.current).toMatchObject({ gpu: "A10G", modelDtype: "bfloat16", flowSteps: 4, slack: 5 });
    act(() => { result.current.setGpu("A100"); result.current.setModelDtype(""); result.current.setFlowSteps(null); });
    rerender({ policy: "act" });
    expect(result.current).toMatchObject({ gpu: "A10G", modelDtype: "float32", flowSteps: null });
    rerender({ policy: "molmoact2" });
    expect(result.current).toMatchObject({ gpu: "A100", modelDtype: "", flowSteps: null });
    unmount();
    const restored = renderHook(() => useGpuKnobs("molmoact2"));
    expect(restored.result.current).toMatchObject({ gpu: "A100", modelDtype: "", flowSteps: null });
  });


});

// Old defaults must not override the global A10G default after an upgrade.
it.each([undefined, "act", "smolvla", "pi0", "molmoact2"])("defaults %s to A10G and remembers a later override", (policy) => {
  localStorage.setItem("makermodslab.gpuType", "A100");
  localStorage.setItem("makermodslab.gpuType.molmoact2", "A100");
  const first = renderHook(() => useGpuKnobs(policy));
  expect(first.result.current.gpu).toBe("A10G");
  act(() => first.result.current.setGpu("H100"));
  first.unmount();
  const next = renderHook(() => useGpuKnobs(policy));
  expect(next.result.current.gpu).toBe("H100");
});
