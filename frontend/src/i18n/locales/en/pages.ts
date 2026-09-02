export default {
  notFound: {
    // "404" is a numeral and reads the same in both languages.
    message: "Oops! Page not found",
    home: "Return to Home",
  },
  teleop: {
    stoppedWarnTitle: "Teleoperation stopped — check the arm",
    stoppedTitle: "Teleoperation stopped",
    // Client-side fallback only: `data.message` from the backend wins when set.
    releasingFallback:
      "The arm returns to its starting position, then goes limp.",
    checkArmTitle: "Check the arm",
    disconnectedCleanly: "The arm was disconnected cleanly.",
    endedWithWarning: "Teleoperation ended with a cleanup warning",
    failed: "Teleoperation failed",
    armLink: {
      heading: "Remote link & servos",
      loading: "loading",
      remote: "Remote action lane",
      live: "Live",
      simulationOnly: "Simulation only",
      maintenance: "Maintenance writes",
      disabled: "Disabled",
      servoHealth: "Servo health",
      noOwner: "Waiting for bus owner",
      healthy: "{{count}} servos · no faults",
      faults_one: "{{count}} decoded fault",
      faults_other: "{{count}} decoded faults",
    },
  },
} as const;
