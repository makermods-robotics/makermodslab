interface CameraInput {
  requestKey: string;
  display: string;
}

const MAX_EXTRA_VIEWS = 2;

/** Camera cards use robot-record order, never USB discovery order.
 * Named inputs match by name. Anonymous cam<N> inputs use the card at N.
 * A missing or disconnected camera must not shift another view into its slot.
 */
export function policyCameraBindings(
  inputs: CameraInput[],
  cameraNames: string[],
  allowExtraViews: boolean,
): { roles: Record<string, string>; extra: string[] } {
  const roles: Record<string, string> = {};
  const used = new Set<string>();
  for (const input of inputs) {
    const camera = cameraNames.find(
      (name) => name.toLowerCase() === input.display.toLowerCase(),
    );
    if (camera != null) {
      roles[input.requestKey] = camera;
      used.add(camera);
    }
  }
  for (const input of inputs) {
    if (roles[input.requestKey] != null) continue;
    const index = /^cam(0|[1-9]\d*)$/.exec(input.display)?.[1];
    const camera = index == null ? undefined : cameraNames[Number(index)];
    if (camera != null && !used.has(camera)) {
      roles[input.requestKey] = camera;
      used.add(camera);
    }
  }

  const extra: string[] = [];
  // Do not add views to fixed-view models or before required inputs resolve.
  if (!allowExtraViews || inputs.length === 0 || inputs.some((input) => !roles[input.requestKey]))
    return { roles, extra };

  const taken = new Set(inputs.map((input) => input.requestKey));
  let next = 0;
  for (const camera of cameraNames) {
    if (used.has(camera)) continue;
    if (extra.length === MAX_EXTRA_VIEWS) break;
    while (taken.has(`cam${next}`)) next += 1;
    const role = `cam${next++}`;
    roles[role] = camera;
    extra.push(role);
    taken.add(role);
    used.add(camera);
  }
  return { roles, extra };
}
