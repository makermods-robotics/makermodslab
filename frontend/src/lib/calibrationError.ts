import type { TFunction } from "i18next";

/** Keep hardware failures actionable without putting driver dumps in a toast. */
export function calibrationErrorMessage(
  error: string | null | undefined,
  t: TFunction,
): string {
  const message = error?.trim();
  if (!message) return t("robotConfig.calib.toast.failedFallback");
  if (/fault reported|reported fault|motor fault/i.test(message)) {
    return t("robotConfig.calib.toast.motorFault");
  }
  if (
    /no response|not responding|never responded|no reply|handshake failed|timed?\s*out|timeout|unreliable|lost response/i.test(
      message,
    ) &&
    !/waiting for.*pose/i.test(message)
  ) {
    const list = message.match(
      /(?:did not respond|missing motors):\s*\[([^\]]+)\]/i,
    );
    const motor = message.match(
      /(?:motor ['"]([^'"]+)['"]|Servo (\w+) \(id=)/i,
    );
    const joints = list
      ? [...list[1].matchAll(/['"]([^'"]+)['"]/g)].map((match) => match[1])
      : motor
        ? [motor[1] || motor[2]]
        : [];
    if (joints.length) {
      return t("robotConfig.calib.toast.jointsNotResponding", {
        joints: joints
          .slice(0, 3)
          .map((name) => name.replace(/_/g, " "))
          .join(", "),
      });
    }
    return t("robotConfig.calib.toast.notResponding");
  }
  if (/permission denied|access.*denied/i.test(message)) {
    return t("robotConfig.calib.toast.portDenied");
  }
  if (
    /no such file|could not open port|not connected|failed to connect.*bus/i.test(
      message,
    )
  ) {
    return t("robotConfig.calib.toast.disconnected");
  }
  const line = message.split(/\r?\n/)[0];
  return line.length > 160 ? `${line.slice(0, 157)}…` : line;
}
