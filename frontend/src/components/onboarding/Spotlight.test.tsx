import { describe, expect, it } from "vitest";
import { render } from "@testing-library/react";
import { OnboardingProvider } from "@/contexts/OnboardingContext";
import Spotlight from "./Spotlight";

describe("Spotlight", () => {
  it("renders nothing when no tour is active", () => {
    const { container } = render(
      <OnboardingProvider>
        <Spotlight />
      </OnboardingProvider>,
    );
    expect(container).toBeEmptyDOMElement();
  });
});
