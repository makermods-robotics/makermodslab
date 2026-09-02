import { afterEach, describe, expect, it } from "vitest";
import { render, screen, act } from "@testing-library/react";
import i18n from "@/i18n";
import { LANGUAGE_STORAGE_KEY } from "@/i18n/config";
import { LanguageProvider, useLanguage } from "@/contexts/LanguageContext";
import Hero from "@/components/launchpad/Hero";

/**
 * End-to-end-ish check that a language change repaints real UI without a
 * reload, which is the one behaviour the catalog-parity tests cannot prove.
 */
function Switcher() {
  const { language, setLanguage } = useLanguage();
  return (
    <button onClick={() => setLanguage(language === "en" ? "zh-CN" : "en")}>
      toggle
    </button>
  );
}

function renderApp() {
  return render(
    <LanguageProvider>
      <Switcher />
      <Hero search="" onSearchChange={() => {}} />
    </LanguageProvider>,
  );
}

afterEach(async () => {
  // Reset both halves of the language state: i18next's active language and
  // the persisted choice the next LanguageProvider would boot from.
  localStorage.clear();
  await act(async () => {
    await i18n.changeLanguage("en");
  });
  // The provider PERSISTS the choice, so resetting i18next is not enough: the
  // next render would re-detect the stored language and start in Chinese.
  // (Only observable once a localStorage exists in the test env at all — see
  // the polyfill in src/test/setup.ts.)
  localStorage.removeItem(LANGUAGE_STORAGE_KEY);
});

describe("LanguageProvider", () => {
  it("renders English copy by default", () => {
    renderApp();
    expect(screen.getByLabelText("Search skills")).toBeInTheDocument();
    expect(screen.getByPlaceholderText("Clean my desk…")).toBeInTheDocument();
    // The rotating verbs all render (stacked in one grid cell).
    expect(screen.getByText("Run")).toBeInTheDocument();
    expect(screen.getByText("Train")).toBeInTheDocument();
  });

  it("repaints the tree in Chinese without a remount, and updates <html lang>", async () => {
    renderApp();
    await act(async () => {
      screen.getByText("toggle").click();
    });

    expect(screen.getByLabelText("搜索技能")).toBeInTheDocument();
    expect(screen.getByPlaceholderText("整理我的桌面…")).toBeInTheDocument();
    expect(screen.getByText("运行")).toBeInTheDocument();
    expect(screen.getByText("训练")).toBeInTheDocument();
    // No English left behind in the switched subtree.
    expect(screen.queryByLabelText("Search skills")).toBeNull();

    expect(document.documentElement.lang).toBe("zh-CN");
  });

  it("switches back to English", async () => {
    renderApp();
    await act(async () => screen.getByText("toggle").click());
    await act(async () => screen.getByText("toggle").click());
    expect(screen.getByLabelText("Search skills")).toBeInTheDocument();
    expect(document.documentElement.lang).toBe("en");
  });
});
