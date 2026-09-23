import { ThermalReplayStatus as Status } from "@/lib/replayHardwareApi";

export function ThermalReplayStatus({ status }: { status?: Status | null }) {
  if (!status) return null;
  const failed = !["running", "completed", "stopped"].includes(status.result);
  return (
    <section aria-label="Thermal test" className="space-y-2 text-xs">
      <p role={failed ? "alert" : "status"} className={failed ? "font-semibold text-red-600" : "font-medium"}>
        {status.critical ? "CRITICAL — at or above 135°C. " : ""}
        {status.result === "completed" && status.rest_reached ? "PASS — " : ""}
        {status.message}
      </p>
      <p>{Math.floor(status.elapsed_s)} / {status.duration_s} seconds · {status.cycles} completed loops · {status.experiment === "baseline" ? "Baseline" : "Shoulder stiffness −15%"}</p>
      <div className="grid grid-cols-1 gap-1 font-mono sm:grid-cols-2">
        {status.actuators.map((motor) => (
          <div key={motor.actuator} className={`rounded px-2 py-1 ${motor.status === "CRITICAL" ? "bg-red-700 font-bold text-white" : motor.status === "OVERHEATING" ? "bg-red-100 text-red-800" : motor.status !== "OK" ? "bg-amber-100 text-amber-900" : "bg-muted"}`}>
            {motor.actuator.replace("follower.", "")} · {motor.temperature_c?.toFixed(1) ?? "—"}°C · {motor.torque_nm?.toFixed(2) ?? "—"} Nm · {motor.status}
          </div>
        ))}
      </div>
      <p className="break-all text-muted-foreground">Run logs: {status.log_dir}</p>
    </section>
  );
}
