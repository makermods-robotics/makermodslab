# Network and Tailscale policy

Remote teleoperation needs two operator-to-robot flows: TLS WebSocket control
on TCP `7443`, and authenticated latest-value actions on UDP `7444`. Use
different ports only if both role configurations, host firewall rules, and
tailnet policy use the same replacements.

Tailscale provides an encrypted private path, not robot action authorization.
The application still requires a locally opened one-time pairing window,
pinned robot certificate, persistent revocable operator credential,
robot-minted session, and per-session HMAC action key.

## Least-privilege tailnet policy

Tailscale recommends **grants** for new policies. Its current syntax supports
protocol-and-port selectors such as `tcp:443` and `udp:53`; see the official
[grants syntax](https://tailscale.com/docs/reference/syntax/grants) and
[examples](https://tailscale.com/docs/reference/examples/grants).

Merge this bounded example into the existing tailnet policy after replacing
the tag names if they already exist:

```jsonc
{
  "tagOwners": {
    "tag:makermodslab-operator": ["autogroup:admin"],
    "tag:makermodslab-robot": ["autogroup:admin"],
  },
  "grants": [
    {
      "src": ["tag:makermodslab-operator"],
      "dst": ["tag:makermodslab-robot"],
      "ip": ["tcp:7443", "udp:7444"],
    },
  ],
}
```

Assign the operator tag only to the intended operator laptop and the robot tag
only to the intended robot laptop. Do not add the reverse grant. Do not allow
TCP `8000` or `8080`: the MakerMods UI/API remains local to each host.

If the tailnet keeps user-owned devices instead of tags, use exact host aliases
or the narrowest existing group/host selectors. Do not paste an `"ip":["*"]`
example into a production policy. Existing broader grants are additive and can
silently defeat this restriction, so review the complete policy.

## Host firewall

On the robot, allow inbound traffic to TCP `7443` and UDP `7444` only through
the Tailscale interface. Deny those ports on Wi-Fi, Ethernet, public, and
wildcard interfaces. The application also refuses wildcard/public binds, but
the host firewall is an independent layer.

For Linux with UFW, after verifying the interface is literally `tailscale0`:

```bash
ip address show tailscale0
sudo ufw allow in on tailscale0 to any port 7443 proto tcp
sudo ufw allow in on tailscale0 to any port 7444 proto udp
sudo ufw status numbered
```

Do not run those commands when the interface name or firewall manager differs.
Use the operating system or site firewall's equivalent scoped rules and record
them in the worksheet. The operator laptop needs outbound access to both ports,
not inbound application ports.

Tailscale usually needs no manual perimeter-firewall opening. Its official
[firewall guidance](https://tailscale.com/docs/reference/faq/firewall-ports)
describes direct WireGuard, STUN, and DERP requirements. Do not expose MakerMods
ports on the public router to improve a relayed connection.

## Preflight

Run from the operator laptop, replacing the name with the robot's tailnet name:

```bash
tailscale ping ROBOT_TAILNET_NAME
tailscale status
```

`tailscale status` reports whether the current path is direct or relayed. A
relayed path is not automatically unsafe, but its latency/jitter must pass the
commissioned clock, heartbeat, and action-watchdog budgets. Never loosen those
budgets merely to hide an unsuitable path.

Before motion, verify from a third, unauthorized tailnet device that TCP `7443`
and UDP `7444` are not granted. A TCP refusal alone does not test the UDP grant;
use the tailnet policy test tools appropriate to the organization's policy.

## Loss injection

During secured-arm commissioning, disable the operator's Tailscale connection
from its local Tailscale UI while a conservative action stream is active. Do
not disable the robot's interface until a robot-side tester is present with
physical power removal reachable. The robot must stop locally without relying
on the operator UI, browser, WebSocket, UDP, or tailnet to recover.
