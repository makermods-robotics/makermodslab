import { useRobots } from "@/hooks/useRobots";
import SessionCameraGrid from "./SessionCameraGrid";

export default function TeleopCameraPanel() {
  const { selectedRecord } = useRobots();
  return <SessionCameraGrid robot={selectedRecord} />;
}
