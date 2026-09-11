type Request = (url: string, init: RequestInit) => Promise<Response>;

/** Swing identification is explicit; auto detection requires unambiguous roles. */
export async function detectArmPort(
  request: Request,
  baseUrl: string,
  armType: string,
  device: "teleop" | "robot",
  multipleMessage: string,
  method: "swing" | "auto" = device === "teleop" ? "swing" : "auto",
  options: {
    portProbe: boolean;
    leaderKind?: string;
    leaderEnergized?: boolean;
  } = { portProbe: true },
) {
  const post = async (path: string, body: object) => {
    const response = await request(`${baseUrl}/api/v1/${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!response.ok)
      throw new Error(`Port detection failed (${response.status})`);
    return response.json();
  };
  if (!options.portProbe) return post("identify-arm", {});
  const leader = options.leaderKind ? { leader_kind: options.leaderKind } : {};
  if (method === "swing" && !options.leaderEnergized) {
    return post("maker/identify-arm", {
      device_type: device,
      arm_type: armType,
      ...leader,
    });
  }
  const probe = await post("maker/probe-ports", {
    arm_type: armType,
    ...leader,
  });
  if (options.leaderEnergized || probe.fallback === "wiggle") {
    return { success: false, message: probe.message, fallback: "wiggle" };
  }
  const followers: string[] = [...new Set<string>(probe.follower_ports ?? [])];
  const leaders = [...new Set<string>(probe.leader_ports ?? [])];
  if (followers.length > 1 || leaders.length > 1) {
    return { success: false, multiple: true, message: multipleMessage };
  }
  const candidates = device === "teleop" ? leaders : followers;
  if (probe.success && candidates.length === 1) {
    return { success: true, port: candidates[0], message: probe.message };
  }
  return { success: false, message: probe.message };
}
