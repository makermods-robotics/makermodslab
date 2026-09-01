import { afterEach, describe, expect, it } from "vitest";
import i18n from "@/i18n";
import { policyTitle, policyDisplayTitle } from "@/components/launchpad/PolicyCard";
import type { ModelItem } from "@/lib/modelsApi";

const SOCKS = "makermods/act_makermods_sock_2_only_more_orange_2026-07-16_22-14-55";

const model = (over: Partial<ModelItem> = {}): ModelItem => ({
  id: SOCKS,
  name: SOCKS,
  policy_type: "act",
  dataset: null,
  steps: null,
  path: null,
  last_modified: null,
  hf_repo_id: SOCKS,
  source: "hub",
  ...over,
});

afterEach(async () => {
  await i18n.changeLanguage("en");
});

describe("policyTitle (data function — search matches against this)", () => {
  it("stays English no matter the UI language", async () => {
    expect(policyTitle(model())).toBe("Sorting socks");
    await i18n.changeLanguage("zh-CN");
    // The invariant that keeps searching "sock" working while the UI is in
    // Chinese. If this ever tracks the active language, search silently breaks.
    expect(policyTitle(model())).toBe("Sorting socks");
  });

  it("falls back to the repo name segment for uncurated policies", () => {
    expect(policyTitle(model({ id: "someone/my-run", hf_repo_id: "someone/my-run", name: "someone/my-run" })))
      .toBe("my-run");
  });
});

describe("policyDisplayTitle (what the user reads)", () => {
  it("follows the active language", async () => {
    const t = i18n.getFixedT("en");
    expect(policyDisplayTitle(t, model())).toBe("Sorting socks");
    expect(policyDisplayTitle(i18n.getFixedT("zh-CN"), model())).toBe("整理袜子");
  });

  it("leaves uncurated repo-derived names untranslated", () => {
    const m = model({ id: "someone/my-run", hf_repo_id: "someone/my-run", name: "someone/my-run" });
    expect(policyDisplayTitle(i18n.getFixedT("zh-CN"), m)).toBe("my-run");
  });
});
