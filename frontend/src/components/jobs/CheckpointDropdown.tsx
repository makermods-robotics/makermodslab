import React from "react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { JobCheckpoint } from "@/lib/checkpointsApi";
import { cn } from "@/lib/utils";

interface Props {
  checkpoints: JobCheckpoint[];
  /** `ref` of the selected checkpoint. The ref is the checkpoint's unique
   * identity (it encodes the owning repo/path plus the step); the step alone
   * is NOT unique when a card merges checkpoints across a resume lineage —
   * two runs can each have a step 2000. */
  selectedRef: string | null;
  onChange: (ckpt: JobCheckpoint) => void;
  disabled?: boolean;
  placeholder?: string;
  /** Extra classes for the trigger (e.g. `w-full min-w-0` when the dropdown
   * flexes inside a card's single-line action row). */
  className?: string;
  /** Trigger id, so a <Label htmlFor> can be attached when the dropdown is
   * rendered as a labelled form field (the Run panel). */
  id?: string;
}

export const CheckpointDropdown: React.FC<Props> = ({
  checkpoints,
  selectedRef,
  onChange,
  disabled,
  placeholder = "Select checkpoint",
  className,
  id,
}) => {
  // step 0 is the sentinel for an imported single-model checkpoint (lerobot
  // never saves at step 0), so it has no meaningful step number — show
  // "latest" instead. Real training checkpoints keep their step label.
  const labelFor = (step: number) => (step === 0 ? "latest" : `step ${step}`);
  return (
    <Select
      value={selectedRef ?? undefined}
      onValueChange={(ref) => {
        const picked = checkpoints.find((c) => c.ref === ref);
        if (picked) onChange(picked);
      }}
      disabled={disabled || checkpoints.length === 0}
    >
      <SelectTrigger
        id={id}
        className={cn(
          "bg-background border-input h-8 text-xs px-2 w-auto min-w-[110px]",
          className,
        )}
        onClick={(e) => e.stopPropagation()}
      >
        <SelectValue placeholder={placeholder} />
      </SelectTrigger>
      <SelectContent className="bg-popover border-border">
        {checkpoints.map((c) => (
          <SelectItem
            key={c.ref}
            value={c.ref}
            onClick={(e) => e.stopPropagation()}
          >
            {labelFor(c.step)}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
};

export default CheckpointDropdown;
