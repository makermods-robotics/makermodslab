import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { useNavigate } from "react-router-dom";
import {
  ArrowLeft,
  Gamepad2,
  KeyRound,
  Loader2,
  RadioTower,
  RefreshCw,
  Server,
  ShieldAlert,
  ShieldCheck,
  Square,
} from "lucide-react";
import BrandMark from "@/components/BrandMark";
import LanguageSwitcher from "@/components/LanguageSwitcher";
import RemoteCommissioningPanel from "@/components/control/RemoteCommissioningPanel";
import RemoteRuntimeStatusPanel from "@/components/control/RemoteRuntimeStatusPanel";
import RemoteSafetyLimitsFields from "@/components/control/RemoteSafetyLimitsFields";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useApi } from "@/contexts/ApiContext";
import { useToast } from "@/hooks/use-toast";
import { useRobots, type RobotRecord } from "@/hooks/useRobots";
import { ApiError } from "@/lib/apiClient";
import { cn } from "@/lib/utils";
import {
  commissionRemoteFollower,
  disableRemoteTeleoperation,
  enableRemoteTeleoperation,
  getRemoteTeleoperationStatus,
  openRemotePairingWindow,
  pairRemoteOperator,
  recoverRemoteHardware,
  removeRemoteConfiguration,
  remoteRuntimeView,
  revokeRemoteCredential,
  saveRemoteConfiguration,
  sendRemoteBrowserHeartbeat,
  stopRemoteTeleoperation,
  type PairingWindowResponse,
  type ConfirmedRemotePhysicalSafeguards,
  type RemoteOperatorConfiguration,
  type RemoteRobotConfiguration,
  type RemoteRole,
  type RemoteTeleoperationStatus,
} from "@/lib/remoteTeleoperationApi";

type BusyAction =
  | "save"
  | "enable"
  | "disable"
  | "commission"
  | "recover"
  | "remove-config"
  | "pair-window"
  | "pair"
  | "revoke"
  | null;

const defaultRobotConfiguration = (): RemoteRobotConfiguration => ({
  node_id: "",
  robot_name: "",
  bind_address: "",
  control_port: 7443,
  udp_port: 7444,
  tls_certificate_path: "",
  tls_private_key_path: "",
  leader_calibration_id: "",
  leader_calibration_digest: "",
  action_rate_hz: 50,
  action_watchdog_ms: 200,
  first_action_deadline_ms: 1000,
  control_deadline_ms: 1000,
  browser_deadline_ms: 2000,
  max_velocity_per_s: 60,
  max_acceleration_per_s2: 300,
});

const defaultOperatorConfiguration = (): RemoteOperatorConfiguration => ({
  node_id: "",
  robot_id: "",
  leader_robot_name: "",
  control_uri: "",
  certificate_fingerprint: "",
  action_rate_hz: 50,
});

function isRemoteSo101Record(record: RobotRecord): boolean {
  return record.mode === "single" && record.arm_type === "so101";
}

const errorDetail = (error: unknown): string =>
  error instanceof ApiError
    ? error.detail ?? error.message
    : error instanceof Error
      ? error.message
      : String(error);

const SelectField = ({
  id,
  label,
  value,
  onChange,
  records,
  side,
  disabled,
  emptyText,
}: {
  id: string;
  label: string;
  value: string;
  onChange: (value: string) => void;
  records: RobotRecord[];
  side: "leader" | "follower";
  disabled?: boolean;
  emptyText: string;
}) => (
  <div className="space-y-1.5">
    <Label htmlFor={id}>{label}</Label>
    <select
      id={id}
      value={value}
      disabled={disabled || records.length === 0}
      onChange={(event) => onChange(event.target.value)}
      className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
    >
      <option value="">{emptyText}</option>
      {records.map((record) => {
        const calibration =
          side === "leader" ? record.leader_config : record.follower_config;
        return (
          <option key={record.name} value={record.name}>
            {record.name}{calibration ? ` — ${calibration}` : ""}
          </option>
        );
      })}
    </select>
  </div>
);

const PairingPayload = ({
  response,
  onClose,
}: {
  response: PairingWindowResponse;
  onClose: () => void;
}) => {
  const { t } = useTranslation();
  const payload = response.payload ?? response;
  const token = payload.pairing_token;
  return (
    <div className="mt-3 rounded-lg border border-warn/40 bg-warn/10 p-3">
      <div className="flex items-center justify-between gap-3">
        <p className="text-sm font-medium">{t("remoteTeleop.pairing.payload")}</p>
        <Button type="button" size="sm" variant="ghost" onClick={onClose}>
          {t("remoteTeleop.pairing.close")}
        </Button>
      </div>
      <dl className="mt-2 space-y-2 text-xs">
        <div>
          <dt className="text-muted-foreground">
            {payload.robot_address}:{payload.control_port}
          </dt>
          <dd className="mt-0.5 break-all font-mono">
            {payload.certificate_fingerprint ?? "—"}
          </dd>
        </div>
        {token && (
          <div>
            <dt className="text-muted-foreground">
              {t("remoteTeleop.pairing.token")}
            </dt>
            <dd className="mt-0.5 select-all break-all rounded bg-background p-2 font-mono text-sm">
              {token}
            </dd>
          </div>
        )}
      </dl>
      {payload.expires_in_ms != null && (
        <p className="mt-2 text-xs text-muted-foreground">
          {t("remoteTeleop.pairing.expires", {
            seconds: Math.max(0, Math.ceil(payload.expires_in_ms / 1000)),
          })}
        </p>
      )}
    </div>
  );
};

export default function RemoteTeleoperationPage() {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const { baseUrl, fetchWithHeaders } = useApi();
  const { toast } = useToast();
  const { records } = useRobots();
  const [role, setRole] = useState<RemoteRole | null>(null);
  const [robotConfig, setRobotConfig] = useState(defaultRobotConfiguration);
  const [operatorConfig, setOperatorConfig] = useState(
    defaultOperatorConfiguration,
  );
  const [status, setStatus] = useState<RemoteTeleoperationStatus | null>(null);
  const [loadingStatus, setLoadingStatus] = useState(true);
  const [busy, setBusy] = useState<BusyAction>(null);
  const [stopping, setStopping] = useState(false);
  const [pairingWindow, setPairingWindow] =
    useState<PairingWindowResponse | null>(null);
  const [pairingToken, setPairingToken] = useState("");
  const [operatorLabel, setOperatorLabel] = useState("");
  const hydratedRef = useRef(false);
  const stoppingRef = useRef(false);

  const remoteRecords = useMemo(
    () => Object.values(records).filter(isRemoteSo101Record),
    [records],
  );
  const selectedFollower = records[robotConfig.robot_name];
  const selectedLeader = records[operatorConfig.leader_robot_name];
  const runtime = remoteRuntimeView(status);
  const robotCredentials = status?.runtime?.credentials ?? [];
  const configuredForRole = Boolean(
    role && status?.configured && status.role === role,
  );
  const robotCanEnable =
    role !== "robot" ||
    Boolean(
      status?.commissioning?.commissioned &&
        !status?.durable_fault?.fault_lockout &&
        !status?.hardware_registry?.held,
    );

  const refreshStatus = useCallback(
    async (signal?: AbortSignal) => {
      try {
        const next = await getRemoteTeleoperationStatus(
          baseUrl,
          fetchWithHeaders,
          signal,
        );
        setStatus(next);
        if (!hydratedRef.current) {
          hydratedRef.current = true;
          if (next.role === "robot" || next.role === "operator") {
            setRole(next.role);
          }
          const config = next.configuration ?? next.config;
          if (next.role === "robot" && config) {
            setRobotConfig((current) => ({
              ...current,
              node_id: config.node_id ?? current.node_id,
              robot_name: config.robot_name ?? current.robot_name,
              bind_address: config.bind_address ?? current.bind_address,
              control_port: config.control_port ?? current.control_port,
              udp_port: config.udp_port ?? current.udp_port,
              leader_calibration_id:
                config.leader_calibration_id ?? current.leader_calibration_id,
              leader_calibration_digest:
                config.leader_calibration_digest ??
                current.leader_calibration_digest,
              action_rate_hz: config.action_rate_hz ?? current.action_rate_hz,
              action_watchdog_ms:
                config.action_watchdog_ms ?? current.action_watchdog_ms,
              first_action_deadline_ms:
                config.first_action_deadline_ms ??
                current.first_action_deadline_ms,
              control_deadline_ms:
                config.control_deadline_ms ?? current.control_deadline_ms,
              browser_deadline_ms:
                config.browser_deadline_ms ?? current.browser_deadline_ms,
              max_velocity_per_s:
                config.max_velocity_per_s ?? current.max_velocity_per_s,
              max_acceleration_per_s2:
                config.max_acceleration_per_s2 ??
                current.max_acceleration_per_s2,
            }));
          }
          if (next.role === "operator" && config) {
            setOperatorConfig((current) => ({
              ...current,
              node_id: config.node_id ?? current.node_id,
              robot_id: config.robot_id ?? current.robot_id,
              leader_robot_name:
                config.leader_robot_name ?? current.leader_robot_name,
              control_uri: config.control_uri ?? current.control_uri,
              certificate_fingerprint:
                config.certificate_fingerprint ??
                current.certificate_fingerprint,
              action_rate_hz: config.action_rate_hz ?? current.action_rate_hz,
            }));
          }
        }
      } catch (error) {
        if (!(error instanceof DOMException && error.name === "AbortError")) {
          setStatus(null);
        }
      } finally {
        setLoadingStatus(false);
      }
    },
    [baseUrl, fetchWithHeaders],
  );

  useEffect(() => {
    const controller = new AbortController();
    void refreshStatus(controller.signal);
    const interval = window.setInterval(() => {
      void refreshStatus();
    }, 1000);
    return () => {
      controller.abort();
      window.clearInterval(interval);
    };
  }, [refreshStatus]);

  // Only an explicitly enabled operator role sends browser liveness. Losing
  // this tab therefore propagates to the robot-local watchdog instead of
  // relying on an unload request that the browser may never deliver.
  useEffect(() => {
    if (role !== "operator" || !runtime.enabled) return;
    const heartbeat = () => {
      void sendRemoteBrowserHeartbeat(baseUrl, fetchWithHeaders).catch(() => {
        // Robot-side deadlines are authoritative; the status poll will expose
        // the resulting stop/fault without logging request or credential data.
      });
    };
    heartbeat();
    const interval = window.setInterval(heartbeat, 750);
    return () => window.clearInterval(interval);
  }, [baseUrl, fetchWithHeaders, role, runtime.enabled]);

  const showError = useCallback(
    (title: string, error: unknown) => {
      toast({
        title,
        description: errorDetail(error),
        variant: "destructive",
      });
    },
    [toast],
  );

  const saveConfiguration = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!role) return;
    setBusy("save");
    try {
      await saveRemoteConfiguration(
        baseUrl,
        fetchWithHeaders,
        role === "robot"
          ? { role, robot: robotConfig, operator: null }
          : { role, robot: null, operator: operatorConfig },
      );
      // Paths are intentionally never retained after submission. Their saved
      // state is represented only by backend booleans in the next status.
      if (role === "robot") {
        setRobotConfig((current) => ({
          ...current,
          tls_certificate_path: "",
          tls_private_key_path: "",
        }));
      }
      toast({
        title: t("remoteTeleop.configuration.savedTitle"),
        description: t("remoteTeleop.configuration.savedDescription"),
      });
      await refreshStatus();
    } catch (error) {
      showError(t("remoteTeleop.configuration.failedTitle"), error);
    } finally {
      setBusy(null);
    }
  };

  const enable = async () => {
    setBusy("enable");
    try {
      await enableRemoteTeleoperation(baseUrl, fetchWithHeaders);
      toast({ title: t("remoteTeleop.controls.enabledTitle") });
      await refreshStatus();
    } catch (error) {
      showError(t("remoteTeleop.controls.failedTitle"), error);
    } finally {
      setBusy(null);
    }
  };

  const commission = async (
    safeguards: ConfirmedRemotePhysicalSafeguards,
  ) => {
    setBusy("commission");
    try {
      await commissionRemoteFollower(baseUrl, fetchWithHeaders, safeguards);
      toast({
        title: t("remoteTeleop.commissioning.commissionedTitle"),
        description: t("remoteTeleop.commissioning.commissionedDescription"),
      });
      await refreshStatus();
    } catch (error) {
      showError(t("remoteTeleop.commissioning.failedTitle"), error);
      await refreshStatus();
    } finally {
      setBusy(null);
    }
  };

  const recoverHardware = async (
    safeguards: ConfirmedRemotePhysicalSafeguards,
  ) => {
    setBusy("recover");
    try {
      await recoverRemoteHardware(baseUrl, fetchWithHeaders, safeguards);
      toast({
        title: t("remoteTeleop.commissioning.recoveredTitle"),
        description: t("remoteTeleop.commissioning.recoveredDescription"),
      });
      await refreshStatus();
    } catch (error) {
      showError(t("remoteTeleop.commissioning.failedTitle"), error);
      await refreshStatus();
    } finally {
      setBusy(null);
    }
  };

  const disable = async () => {
    setBusy("disable");
    try {
      await disableRemoteTeleoperation(baseUrl, fetchWithHeaders);
      toast({ title: t("remoteTeleop.controls.disabledTitle") });
      await refreshStatus();
    } catch (error) {
      showError(t("remoteTeleop.controls.failedTitle"), error);
    } finally {
      setBusy(null);
    }
  };

  const removeConfiguration = async () => {
    if (!window.confirm(t("remoteTeleop.controls.removeConfirm"))) return;
    setBusy("remove-config");
    try {
      const next = await removeRemoteConfiguration(baseUrl, fetchWithHeaders);
      setStatus(next);
      toast({
        title: t("remoteTeleop.controls.removedTitle"),
        description: t("remoteTeleop.controls.removedDescription"),
      });
    } catch (error) {
      showError(t("remoteTeleop.controls.removeFailedTitle"), error);
    } finally {
      setBusy(null);
    }
  };

  const stop = useCallback(async () => {
    if (stoppingRef.current) return;
    stoppingRef.current = true;
    setStopping(true);
    try {
      await stopRemoteTeleoperation(baseUrl, fetchWithHeaders);
      toast({
        title: t("remoteTeleop.stop.title"),
        description: t("remoteTeleop.stop.description"),
      });
      await refreshStatus();
    } catch (error) {
      showError(t("remoteTeleop.stop.failedTitle"), error);
    } finally {
      stoppingRef.current = false;
      setStopping(false);
    }
  }, [baseUrl, fetchWithHeaders, refreshStatus, showError, t, toast]);

  const liveState =
    runtime.enabled ||
    !["disabled", "idle", "configured", "simulation-ready"].includes(
      runtime.state.toLowerCase(),
    );
  useEffect(() => {
    if (!liveState) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape" && !event.repeat) {
        event.preventDefault();
        void stop();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [liveState, stop]);

  const openPairingWindow = async () => {
    setBusy("pair-window");
    setPairingWindow(null);
    try {
      const response = await openRemotePairingWindow(baseUrl, fetchWithHeaders);
      setPairingWindow(response);
    } catch (error) {
      showError(t("remoteTeleop.pairing.failedTitle"), error);
    } finally {
      setBusy(null);
    }
  };

  const pair = async (event: React.FormEvent) => {
    event.preventDefault();
    const token = pairingToken.trim();
    const label = operatorLabel.trim();
    if (!token || !label) return;
    // Clear the one-time secret before awaiting network or UI work. It is sent
    // only in the POST body and is never written to storage or a log.
    setPairingToken("");
    setBusy("pair");
    try {
      const response = await pairRemoteOperator(baseUrl, fetchWithHeaders, {
        pairing_token: token,
        operator_label: label,
      });
      toast({
        title: t("remoteTeleop.pairing.pairedTitle"),
        description: t("remoteTeleop.pairing.pairedDescription", {
          credential: response.credential_id,
        }),
      });
      await refreshStatus();
    } catch (error) {
      showError(t("remoteTeleop.pairing.failedTitle"), error);
    } finally {
      setBusy(null);
    }
  };

  const revoke = async (credentialId: string) => {
    setBusy("revoke");
    try {
      await revokeRemoteCredential(
        baseUrl,
        fetchWithHeaders,
        credentialId,
      );
      toast({ title: t("remoteTeleop.pairing.revokedTitle") });
      await refreshStatus();
    } catch (error) {
      showError(t("remoteTeleop.pairing.failedTitle"), error);
    } finally {
      setBusy(null);
    }
  };

  return (
    <div className="min-h-screen bg-background text-foreground">
      <header className="sticky top-0 z-30 border-b border-border bg-background/95 backdrop-blur">
        <div className="mx-auto flex max-w-6xl items-center justify-between gap-3 px-4 py-3 sm:px-6">
          <div className="flex items-center gap-3">
            <Button
              variant="ghost"
              size="sm"
              className="gap-1.5"
              onClick={() => navigate("/")}
            >
              <ArrowLeft className="h-4 w-4" />
              <span className="hidden sm:inline">{t("remoteTeleop.back")}</span>
            </Button>
            <BrandMark />
          </div>
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              className="gap-1.5"
              onClick={() => void refreshStatus()}
              disabled={loadingStatus}
            >
              <RefreshCw
                className={cn("h-3.5 w-3.5", loadingStatus && "animate-spin")}
              />
              <span className="hidden sm:inline">{t("remoteTeleop.refresh")}</span>
            </Button>
            <LanguageSwitcher />
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-6xl px-4 py-6 sm:px-6 sm:py-8">
        <div className="mb-6">
          <h1 className="text-2xl font-semibold tracking-tight sm:text-3xl">
            {t("remoteTeleop.title")}
          </h1>
          <p className="mt-1 text-sm text-muted-foreground">
            {t("remoteTeleop.subtitle")}
          </p>
        </div>

        <section>
          <h2 className="text-sm font-semibold">
            {t("remoteTeleop.role.heading")}
          </h2>
          <div className="mt-2 grid gap-3 sm:grid-cols-2">
            {(["robot", "operator"] as const).map((candidate) => {
              const selected = role === candidate;
              const Icon = candidate === "robot" ? Server : Gamepad2;
              return (
                <button
                  key={candidate}
                  type="button"
                  disabled={runtime.enabled}
                  onClick={() => setRole(candidate)}
                  className={cn(
                    "flex items-start gap-3 rounded-xl border bg-card p-4 text-left transition-colors",
                    selected
                      ? "border-primary ring-1 ring-primary"
                      : "border-border hover:border-primary/40",
                    runtime.enabled && "cursor-not-allowed opacity-60",
                  )}
                >
                  <Icon className="mt-0.5 h-5 w-5 text-primary" />
                  <span>
                    <span className="block text-sm font-semibold">
                      {t(`remoteTeleop.role.${candidate}`)}
                    </span>
                    <span className="mt-1 block text-xs text-muted-foreground">
                      {t(`remoteTeleop.role.${candidate}Description`)}
                    </span>
                  </span>
                </button>
              );
            })}
          </div>
        </section>

        <div className="mt-6 grid items-start gap-4 lg:grid-cols-[minmax(0,1.08fr)_minmax(360px,0.92fr)]">
          <div className="space-y-4">
            <section className="rounded-xl border border-border bg-card p-4 shadow-1">
              <div className="flex items-center justify-between gap-3">
                <h2 className="text-base font-semibold">
                  {t("remoteTeleop.configuration.heading")}
                </h2>
                <Badge variant={configuredForRole ? "secondary" : "outline"}>
                  {configuredForRole
                    ? t("remoteTeleop.configuration.configured")
                    : t("remoteTeleop.configuration.notConfigured")}
                </Badge>
              </div>

              {!role ? (
                <p className="mt-4 text-sm text-muted-foreground">
                  {t("remoteTeleop.role.heading")}
                </p>
              ) : (
                <form onSubmit={saveConfiguration} className="mt-4 space-y-4">
                  {role === "robot" ? (
                    <>
                      <div className="grid gap-4 sm:grid-cols-2">
                        <div className="space-y-1.5">
                          <Label htmlFor="remote-robot-node-id">
                            {t("remoteTeleop.configuration.nodeId")}
                          </Label>
                          <Input
                            id="remote-robot-node-id"
                            value={robotConfig.node_id}
                            onChange={(event) =>
                              setRobotConfig((current) => ({
                                ...current,
                                node_id: event.target.value,
                              }))
                            }
                            placeholder={t(
                              "remoteTeleop.configuration.robotNodeIdPlaceholder",
                            )}
                            required
                          />
                        </div>
                        <SelectField
                          id="remote-follower-record"
                          label={t("remoteTeleop.configuration.follower")}
                          value={robotConfig.robot_name}
                          onChange={(robot_name) =>
                            setRobotConfig((current) => ({
                              ...current,
                              robot_name,
                            }))
                          }
                          records={remoteRecords}
                          side="follower"
                          emptyText={t("remoteTeleop.configuration.noSo101")}
                        />
                      </div>
                      {selectedFollower && (
                        <p className="text-xs text-muted-foreground">
                          {t("remoteTeleop.configuration.savedCalibration", {
                            calibration:
                              selectedFollower.follower_config || "—",
                          })}
                        </p>
                      )}
                      <div className="grid gap-4 sm:grid-cols-3">
                        <div className="space-y-1.5 sm:col-span-1">
                          <Label htmlFor="remote-bind-address">
                            {t("remoteTeleop.configuration.bindAddress")}
                          </Label>
                          <Input
                            id="remote-bind-address"
                            value={robotConfig.bind_address}
                            onChange={(event) =>
                              setRobotConfig((current) => ({
                                ...current,
                                bind_address: event.target.value,
                              }))
                            }
                            placeholder={t(
                              "remoteTeleop.configuration.bindAddressPlaceholder",
                            )}
                            inputMode="decimal"
                            required
                          />
                        </div>
                        <div className="space-y-1.5">
                          <Label htmlFor="remote-control-port">
                            {t("remoteTeleop.configuration.controlPort")}
                          </Label>
                          <Input
                            id="remote-control-port"
                            type="number"
                            min={1}
                            max={65535}
                            value={robotConfig.control_port}
                            onChange={(event) =>
                              setRobotConfig((current) => ({
                                ...current,
                                control_port: Number(event.target.value),
                              }))
                            }
                            required
                          />
                        </div>
                        <div className="space-y-1.5">
                          <Label htmlFor="remote-udp-port">
                            {t("remoteTeleop.configuration.udpPort")}
                          </Label>
                          <Input
                            id="remote-udp-port"
                            type="number"
                            min={1}
                            max={65535}
                            value={robotConfig.udp_port}
                            onChange={(event) =>
                              setRobotConfig((current) => ({
                                ...current,
                                udp_port: Number(event.target.value),
                              }))
                            }
                            required
                          />
                        </div>
                      </div>
                      <div className="grid gap-4 sm:grid-cols-2">
                        <div className="space-y-1.5">
                          <Label htmlFor="remote-tls-certificate">
                            {t("remoteTeleop.configuration.tlsCertificate")}
                          </Label>
                          <Input
                            id="remote-tls-certificate"
                            value={robotConfig.tls_certificate_path}
                            onChange={(event) =>
                              setRobotConfig((current) => ({
                                ...current,
                                tls_certificate_path: event.target.value,
                              }))
                            }
                            placeholder={t(
                              "remoteTeleop.configuration.pathPlaceholder",
                            )}
                            autoComplete="off"
                            spellCheck={false}
                            required
                          />
                        </div>
                        <div className="space-y-1.5">
                          <Label htmlFor="remote-tls-private-key">
                            {t("remoteTeleop.configuration.tlsPrivateKey")}
                          </Label>
                          <Input
                            id="remote-tls-private-key"
                            type="password"
                            value={robotConfig.tls_private_key_path}
                            onChange={(event) =>
                              setRobotConfig((current) => ({
                                ...current,
                                tls_private_key_path: event.target.value,
                              }))
                            }
                            placeholder={t(
                              "remoteTeleop.configuration.pathPlaceholder",
                            )}
                            autoComplete="new-password"
                            spellCheck={false}
                            required
                          />
                        </div>
                      </div>
                      <div className="grid gap-4 sm:grid-cols-2">
                        <div className="space-y-1.5">
                          <Label htmlFor="remote-allowed-calibration-id">
                            {t(
                              "remoteTeleop.configuration.allowedLeaderCalibrationId",
                            )}
                          </Label>
                          <Input
                            id="remote-allowed-calibration-id"
                            value={robotConfig.leader_calibration_id}
                            onChange={(event) =>
                              setRobotConfig((current) => ({
                                ...current,
                                leader_calibration_id: event.target.value,
                              }))
                            }
                            required
                          />
                        </div>
                        <div className="space-y-1.5">
                          <Label htmlFor="remote-allowed-calibration-digest">
                            {t(
                              "remoteTeleop.configuration.allowedLeaderCalibrationDigest",
                            )}
                          </Label>
                          <Input
                            id="remote-allowed-calibration-digest"
                            value={robotConfig.leader_calibration_digest}
                            onChange={(event) =>
                              setRobotConfig((current) => ({
                                ...current,
                                leader_calibration_digest: event.target.value,
                              }))
                            }
                            placeholder={t(
                              "remoteTeleop.configuration.digestPlaceholder",
                            )}
                            pattern="[0-9a-f]{64}"
                            minLength={64}
                            maxLength={64}
                            autoComplete="off"
                            spellCheck={false}
                            required
                          />
                        </div>
                      </div>
                      <RemoteSafetyLimitsFields
                        value={robotConfig}
                        onChange={(limits) =>
                          setRobotConfig((current) => ({
                            ...current,
                            ...limits,
                          }))
                        }
                        disabled={busy !== null || stopping}
                      />
                    </>
                  ) : (
                    <>
                      <div className="grid gap-4 sm:grid-cols-2">
                        <div className="space-y-1.5">
                          <Label htmlFor="remote-operator-node-id">
                            {t("remoteTeleop.configuration.nodeId")}
                          </Label>
                          <Input
                            id="remote-operator-node-id"
                            value={operatorConfig.node_id}
                            onChange={(event) =>
                              setOperatorConfig((current) => ({
                                ...current,
                                node_id: event.target.value,
                              }))
                            }
                            placeholder={t(
                              "remoteTeleop.configuration.operatorNodeIdPlaceholder",
                            )}
                            required
                          />
                        </div>
                        <SelectField
                          id="remote-leader-record"
                          label={t("remoteTeleop.configuration.leader")}
                          value={operatorConfig.leader_robot_name}
                          onChange={(leader_robot_name) =>
                            setOperatorConfig((current) => ({
                              ...current,
                              leader_robot_name,
                            }))
                          }
                          records={remoteRecords}
                          side="leader"
                          emptyText={t("remoteTeleop.configuration.noSo101")}
                        />
                      </div>
                      {selectedLeader && (
                        <p className="text-xs text-muted-foreground">
                          {t("remoteTeleop.configuration.savedCalibration", {
                            calibration: selectedLeader.leader_config || "—",
                          })}
                        </p>
                      )}
                      <div className="grid gap-4 sm:grid-cols-2">
                        <div className="space-y-1.5">
                          <Label htmlFor="remote-robot-id">
                            {t("remoteTeleop.configuration.robotId")}
                          </Label>
                          <Input
                            id="remote-robot-id"
                            value={operatorConfig.robot_id}
                            onChange={(event) =>
                              setOperatorConfig((current) => ({
                                ...current,
                                robot_id: event.target.value,
                              }))
                            }
                            placeholder={t(
                              "remoteTeleop.configuration.robotIdPlaceholder",
                            )}
                            required
                          />
                        </div>
                        <div className="space-y-1.5">
                          <Label htmlFor="remote-control-uri">
                            {t("remoteTeleop.configuration.robotAddress")}
                          </Label>
                          <Input
                            id="remote-control-uri"
                            type="url"
                            value={operatorConfig.control_uri}
                            onChange={(event) =>
                              setOperatorConfig((current) => ({
                                ...current,
                                control_uri: event.target.value,
                              }))
                            }
                            placeholder={t(
                              "remoteTeleop.configuration.robotAddressPlaceholder",
                            )}
                            autoComplete="off"
                            spellCheck={false}
                            required
                          />
                        </div>
                      </div>
                      <div className="space-y-1.5">
                        <Label htmlFor="remote-certificate-fingerprint">
                          {t(
                            "remoteTeleop.configuration.certificateFingerprint",
                          )}
                        </Label>
                        <Input
                          id="remote-certificate-fingerprint"
                          value={operatorConfig.certificate_fingerprint}
                          onChange={(event) =>
                            setOperatorConfig((current) => ({
                              ...current,
                              certificate_fingerprint: event.target.value,
                            }))
                          }
                          placeholder={t(
                            "remoteTeleop.configuration.fingerprintPlaceholder",
                          )}
                          pattern="(?:[0-9A-Fa-f]{2}:?){32}"
                          autoComplete="off"
                          spellCheck={false}
                          required
                        />
                      </div>
                    </>
                  )}

                  <div className="grid gap-4 sm:grid-cols-[160px_1fr] sm:items-end">
                    <div className="space-y-1.5">
                      <Label htmlFor="remote-action-rate">
                        {t("remoteTeleop.configuration.actionRate")}
                      </Label>
                      <Input
                        id="remote-action-rate"
                        type="number"
                        min={10}
                        max={100}
                        value={
                          role === "robot"
                            ? robotConfig.action_rate_hz
                            : operatorConfig.action_rate_hz
                        }
                        onChange={(event) => {
                          const action_rate_hz = Number(event.target.value);
                          if (role === "robot") {
                            setRobotConfig((current) => ({
                              ...current,
                              action_rate_hz,
                            }));
                          } else {
                            setOperatorConfig((current) => ({
                              ...current,
                              action_rate_hz,
                            }));
                          }
                        }}
                        required
                      />
                    </div>
                    <Button
                      type="submit"
                      disabled={
                        busy !== null || stopping || remoteRecords.length === 0
                      }
                    >
                      {busy === "save" ? (
                        <>
                          <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                          {t("remoteTeleop.configuration.saving")}
                        </>
                      ) : (
                        t("remoteTeleop.configuration.save")
                      )}
                    </Button>
                  </div>
                  <p className="flex items-start gap-2 text-xs text-muted-foreground">
                    <ShieldCheck className="mt-0.5 h-3.5 w-3.5 shrink-0" />
                    {t("remoteTeleop.configuration.secretNotice")}
                  </p>
                </form>
              )}
            </section>

            {role === "robot" && (
              <RemoteCommissioningPanel
                status={status}
                busyAction={
                  busy === "commission" || busy === "recover" ? busy : null
                }
                onCommission={commission}
                onRecover={recoverHardware}
              />
            )}

            {role && (
              <section className="rounded-xl border border-border bg-card p-4 shadow-1">
                <div>
                  <h2 className="text-base font-semibold">
                    {t("remoteTeleop.controls.heading")}
                  </h2>
                  <p className="mt-0.5 text-xs text-muted-foreground">
                    {t("remoteTeleop.controls.description")}
                  </p>
                </div>
                <div className="mt-4 flex flex-wrap gap-2">
                  <Button
                    type="button"
                    className="gap-2"
                    onClick={enable}
                    disabled={
                      !configuredForRole ||
                      runtime.enabled ||
                      !robotCanEnable ||
                      busy !== null ||
                      stopping
                    }
                  >
                    {busy === "enable" ? (
                      <Loader2 className="h-4 w-4 animate-spin" />
                    ) : (
                      <RadioTower className="h-4 w-4" />
                    )}
                    {busy === "enable"
                      ? t("remoteTeleop.controls.enabling")
                      : role === "robot"
                        ? t("remoteTeleop.controls.enableRobot")
                        : t("remoteTeleop.controls.enableOperator")}
                  </Button>
                  <Button
                    type="button"
                    variant="outline"
                    onClick={disable}
                    disabled={!runtime.enabled || busy !== null || stopping}
                  >
                    {busy === "disable"
                      ? t("remoteTeleop.controls.disabling")
                      : t("remoteTeleop.controls.disable")}
                  </Button>
                  <Button
                    type="button"
                    variant="ghost"
                    className="text-destructive hover:text-destructive"
                    onClick={() => void removeConfiguration()}
                    disabled={
                      !configuredForRole ||
                      runtime.enabled ||
                      busy !== null ||
                      stopping ||
                      Boolean(status?.durable_fault?.fault_lockout)
                    }
                  >
                    {busy === "remove-config"
                      ? t("remoteTeleop.controls.removing")
                      : t("remoteTeleop.controls.removeConfiguration")}
                  </Button>
                </div>
              </section>
            )}

            {role === "robot" && (
              <section className="rounded-xl border border-border bg-card p-4 shadow-1">
                <h2 className="flex items-center gap-2 text-base font-semibold">
                  <KeyRound className="h-4 w-4" />
                  {t("remoteTeleop.pairing.robotHeading")}
                </h2>
                <p className="mt-1 text-xs text-muted-foreground">
                  {t("remoteTeleop.pairing.robotDescription")}
                </p>
                <Button
                  type="button"
                  variant="outline"
                  className="mt-3"
                  onClick={openPairingWindow}
                  disabled={
                    !configuredForRole ||
                    !runtime.enabled ||
                    busy !== null ||
                    stopping
                  }
                >
                  {busy === "pair-window"
                    ? t("remoteTeleop.pairing.opening")
                    : t("remoteTeleop.pairing.openWindow")}
                </Button>
                {pairingWindow && (
                  <PairingPayload
                    response={pairingWindow}
                    onClose={() => setPairingWindow(null)}
                  />
                )}
                {robotCredentials.length > 0 && (
                  <div className="mt-4 border-t border-border pt-3">
                    <p className="text-xs font-medium text-muted-foreground">
                      {t("remoteTeleop.pairing.credentialsHeading")}
                    </p>
                    <div className="mt-2 space-y-2">
                      {robotCredentials.map((credential) => (
                        <div
                          key={credential.credential_id}
                          className="flex flex-wrap items-center justify-between gap-2 rounded-lg border border-border bg-background/60 p-2.5"
                        >
                          <div className="min-w-0">
                            <p className="truncate text-sm font-medium">
                              {credential.operator_label}
                            </p>
                            <p className="truncate font-mono text-[11px] text-muted-foreground">
                              {credential.credential_id}
                            </p>
                          </div>
                          {credential.revoked ? (
                            <Badge variant="outline">
                              {t("remoteTeleop.pairing.revokedCredential")}
                            </Badge>
                          ) : (
                            <Button
                              type="button"
                              size="sm"
                              variant="destructive"
                              disabled={busy !== null || stopping}
                              onClick={() =>
                                void revoke(credential.credential_id)
                              }
                            >
                              {busy === "revoke"
                                ? t("remoteTeleop.pairing.revoking")
                                : t("remoteTeleop.pairing.revokeCredential")}
                            </Button>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </section>
            )}

            {role === "operator" && (
              <section className="rounded-xl border border-border bg-card p-4 shadow-1">
                <h2 className="flex items-center gap-2 text-base font-semibold">
                  <KeyRound className="h-4 w-4" />
                  {t("remoteTeleop.pairing.operatorHeading")}
                </h2>
                <p className="mt-1 text-xs text-muted-foreground">
                  {t("remoteTeleop.pairing.operatorDescription")}
                </p>
                <form onSubmit={pair} className="mt-3 space-y-3">
                  <div className="space-y-1.5">
                    <Label htmlFor="remote-pairing-token">
                      {t("remoteTeleop.pairing.token")}
                    </Label>
                    <Input
                      id="remote-pairing-token"
                      type="password"
                      value={pairingToken}
                      onChange={(event) => setPairingToken(event.target.value)}
                      placeholder={t("remoteTeleop.pairing.tokenPlaceholder")}
                      autoComplete="new-password"
                      spellCheck={false}
                      required
                    />
                  </div>
                  <div className="space-y-1.5">
                    <Label htmlFor="remote-operator-label">
                      {t("remoteTeleop.pairing.operatorLabel")}
                    </Label>
                    <Input
                      id="remote-operator-label"
                      value={operatorLabel}
                      onChange={(event) => setOperatorLabel(event.target.value)}
                      placeholder={t(
                        "remoteTeleop.pairing.operatorLabelPlaceholder",
                      )}
                      required
                    />
                  </div>
                  <div className="flex flex-wrap gap-2">
                    <Button
                      type="submit"
                      disabled={!configuredForRole || busy !== null || stopping}
                    >
                      {busy === "pair"
                        ? t("remoteTeleop.pairing.pairing")
                        : t("remoteTeleop.pairing.pair")}
                    </Button>
                  </div>
                </form>
              </section>
            )}
          </div>

          <div className="lg:sticky lg:top-20">
            <RemoteRuntimeStatusPanel status={status} loading={loadingStatus} />
            {runtime.faultLockout && (
              <div className="mt-3 flex gap-2 rounded-xl border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive">
                <ShieldAlert className="mt-0.5 h-4 w-4 shrink-0" />
                {t("remoteTeleop.status.locked")}
              </div>
            )}
          </div>
        </div>
      </main>

      <Button
        type="button"
        size="lg"
        variant="destructive"
        onClick={() => void stop()}
        disabled={stopping}
        aria-keyshortcuts="Escape"
        className="fixed bottom-5 right-5 z-50 h-14 min-w-36 gap-2 rounded-full px-6 text-base font-bold shadow-lg sm:bottom-7 sm:right-7"
      >
        {stopping ? (
          <Loader2 className="h-5 w-5 animate-spin" />
        ) : (
          <Square className="h-5 w-5 fill-current" />
        )}
        {stopping
          ? t("remoteTeleop.stop.stopping")
          : t("remoteTeleop.stop.button")}
        <kbd className="ml-1 rounded border border-destructive-foreground/30 px-1.5 py-0.5 text-[10px] font-normal">
          {t("remoteTeleop.stop.shortcut")}
        </kbd>
      </Button>
    </div>
  );
}
