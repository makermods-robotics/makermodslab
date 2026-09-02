import { BrowserRouter, Routes, Route } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ThemeProvider } from "@/contexts/ThemeContext";
import { LanguageProvider } from "@/contexts/LanguageContext";
import { UrdfProvider } from "@/contexts/UrdfContext";
import { DragAndDropProvider } from "@/contexts/DragAndDropContext";
import { Toaster } from "@/components/ui/toaster";
import { StudioProvider } from "@/contexts/StudioContext";
import { InferenceSessionProvider } from "@/contexts/InferenceSessionContext";
import { OnboardingProvider } from "@/contexts/OnboardingContext";
import Spotlight from "@/components/onboarding/Spotlight";
import Launchpad from "@/pages/Launchpad";
import Teleoperation from "@/pages/Teleoperation";
import RemoteTeleoperation from "@/pages/RemoteTeleoperation";
import Training from "@/pages/Training";
import NotFound from "@/pages/NotFound";
import UpdateNotice from "@/components/UpdateNotice";
import MockHubBanner from "@/components/MockHubBanner";
import { TooltipProvider } from "@radix-ui/react-tooltip";
import { ApiProvider } from "./contexts/ApiContext";
import { HfAuthProvider } from "./contexts/HfAuthContext";
import { SessionProvider } from "./contexts/SessionContext";

const queryClient = new QueryClient();

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <ThemeProvider>
          <LanguageProvider>
          <ApiProvider>
           <SessionProvider>
            <HfAuthProvider>
              <UrdfProvider>
                <DragAndDropProvider>
                  <BrowserRouter>
                    <StudioProvider>
                     <InferenceSessionProvider>
                      <OnboardingProvider>
                        {/* SingleTabGuard used to live here. Hardware
                            exclusivity is now server-authoritative (409
                            session.held names the holder), and the unload
                            beacons whose crossfire the guard prevented — one
                            tab's unload stopping another tab's session — are
                            gone with it: sessions carry a server-side lease
                            instead. Multiple tabs are simply allowed now. */}
                        <UpdateNotice />
                        <MockHubBanner />
                        <Routes>
                          <Route path="/" element={<Launchpad />} />
                          <Route path="/teleoperation" element={<Teleoperation />} />
                          <Route
                            path="/remote-teleoperation"
                            element={<RemoteTeleoperation />}
                          />
                          {/* /training (no id) hosts the shared configurator.
                              Nothing in the app navigates here any more —
                              Continue / Resume and Fine-tune all seed the
                              studio Train panel's in-place form instead — but
                              the route still honours `state.resume` /
                              `state.finetune`, so deep links and stale
                              bookmarks keep working. /training/:jobId is the
                              monitor. */}
                          <Route path="/training" element={<Training />} />
                          <Route path="/training/:jobId" element={<Training />} />
                          {/* /inference is no longer a route — it's the
                              InferenceSessionDialog window, hosted by
                              InferenceSessionProvider and opened by the launch
                              flows (Deploy panel + InferenceModal). */}
                          {/* Robot settings is no longer a route — it's the
                              RobotConfigDialog window, opened from the robot
                              corner (Launchpad + studio headers). */}

                          <Route path="*" element={<NotFound />} />
                        </Routes>
                       <Spotlight />
                      </OnboardingProvider>
                     </InferenceSessionProvider>
                      <Toaster />
                    </StudioProvider>
                  </BrowserRouter>
                </DragAndDropProvider>
              </UrdfProvider>
            </HfAuthProvider>
           </SessionProvider>
          </ApiProvider>
          </LanguageProvider>
        </ThemeProvider>
      </TooltipProvider>
    </QueryClientProvider>
  );
}

export default App;
