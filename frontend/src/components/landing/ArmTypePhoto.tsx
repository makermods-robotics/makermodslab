import React from "react";
import { ImageIcon } from "lucide-react";

/**
 * The picture at the top of an arm-type card in CreateRobotDialog.
 *
 * The frame is `aspect-[4/3]` and always occupies the same box. A placeholder
 * carrying the expected source dimensions remains available for future arm
 * types whose product photo has not landed yet.
 */

/** Source dimensions the placeholder advertises: 4:3, ~2x the largest slot the
 * card ever renders at, so the photo stays crisp on a HiDPI screen. */
export const ARM_PHOTO_SOURCE = { width: 800, height: 600 };

interface ArmTypePhotoProps {
  /** Imported image URL, or null while a hardware photo is unavailable. */
  src: string | null;
  /** Alt text — the arm's own display name, already localized by the caller. */
  alt: string;
}

const ArmTypePhoto: React.FC<ArmTypePhotoProps> = ({ src, alt }) =>
  src ? (
    <img
      src={src}
      alt={alt}
      loading="lazy"
      className="aspect-[4/3] w-full rounded-sm object-cover"
    />
  ) : (
    <div
      // Not `aria-hidden`: an empty frame carries no information a screen
      // reader needs, and the card's label already names the arm.
      aria-hidden="true"
      className="flex aspect-[4/3] w-full flex-col items-center justify-center gap-1 rounded-sm border border-dashed border-border bg-muted/40 text-muted-foreground"
    >
      <ImageIcon className="h-5 w-5" />
      {/* Dimensions, not prose — deliberately unlocalized scaffolding copy that
          goes away with the first real photo. */}
      <span className="font-mono text-[10px] leading-none">
        {ARM_PHOTO_SOURCE.width} × {ARM_PHOTO_SOURCE.height}
      </span>
    </div>
  );

export default ArmTypePhoto;
