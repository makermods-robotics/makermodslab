import { describe, expect, it, vi } from "vitest";
import { LoadingManager, Object3D } from "three";
import { loadMeshFile } from "./meshLoaders";
import { setupMeshLoader, type URDFViewerElement } from "./urdfViewerHelpers";

vi.mock("./meshLoaders", () => ({ loadMeshFile: vi.fn() }));
vi.mock("@/components/ui/sonner", () => ({ toast: {} }));

describe("per-viewer mesh loading", () => {
  it("waits for both arms when Three.js shares an in-flight request", () => {
    const pending: Array<(result: Object3D) => void> = [];
    // Coalesced FileLoader requests skip the second manager's itemStart.
    vi.mocked(loadMeshFile).mockImplementation((_path, _manager, done) => {
      pending.push(done);
    });
    const loaded = [vi.fn(), vi.fn()];
    const attached = [vi.fn(), vi.fn()];
    for (let i = 0; i < 2; i++) {
      const viewer = { loadMeshFunc: undefined } as unknown as URDFViewerElement;
      const manager = new LoadingManager(loaded[i]);
      setupMeshLoader(viewer, (path) => `/metal-urdf/${path}`);
      manager.itemStart("robot.urdf");
      viewer.loadMeshFunc!("meshes/link1.STL", manager, attached[i]);
      manager.itemEnd("robot.urdf");
      expect(loaded[i]).not.toHaveBeenCalled();
    }
    pending[0](new Object3D());
    expect(loaded[0]).toHaveBeenCalledOnce();
    expect(loaded[1]).not.toHaveBeenCalled();
    pending[1](new Object3D());
    expect(attached[1]).toHaveBeenCalledOnce();
    expect(loaded[1]).toHaveBeenCalledOnce();
    expect(attached[1].mock.invocationCallOrder[0]).toBeLessThan(loaded[1].mock.invocationCallOrder[0]);
  });

  it("finishes a failed mesh without leaving the model loading forever", () => {
    const error = new Error("missing mesh");
    vi.mocked(loadMeshFile).mockImplementation(() => { throw error; });
    const loaded = vi.fn();
    const done = vi.fn();
    const viewer = { loadMeshFunc: undefined } as unknown as URDFViewerElement;
    setupMeshLoader(viewer, null);
    viewer.loadMeshFunc!("missing.STL", new LoadingManager(loaded), done);
    expect(done).toHaveBeenCalledWith(null, error);
    expect(loaded).toHaveBeenCalledOnce();
  });
});
