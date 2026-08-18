import React, { useState } from "react";
import { Trans, useTranslation } from "react-i18next";
import { Plus } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { validateDatasetName } from "@/lib/datasetName";
import { useHfAuth } from "@/contexts/HfAuthContext";

interface CreateDatasetDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  /** Existing repo ids, used to warn before offering a name that already
   * exists (matched case-insensitively against the bare name). */
  existingRepoIds: string[];
  /** Called with the validated bare dataset name. The parent seeds the
   * recording modal (which applies any namespace/prefix on start). */
  onCreateNew: (name: string) => void;
}

/**
 * Name-only form for creating a new dataset. Uses the same
 * `validateDatasetName` the picker footer used, so the recorder never receives
 * a name it would later reject. On confirm it hands the bare name to the parent
 * (handleCreateDataset), which opens the recording modal.
 */
/** Placeholder AND live-preview stand-in for an untyped name. A literal repo
 * segment the user could type verbatim — data, so it is not a catalog entry. */
const SAMPLE_NAME = "my_dataset";

const CreateDatasetDialog: React.FC<CreateDatasetDialogProps> = ({
  open,
  onOpenChange,
  existingRepoIds,
  onCreateNew,
}) => {
  const { t } = useTranslation();
  const [name, setName] = useState("");

  // The namespace (`<hf_username>/`) is prepended downstream at record start,
  // so surface it here as a live hint. Only meaningful when the user hasn't
  // typed their own "namespace/name" and we know who they are.
  const { auth } = useHfAuth();
  const username = auth.status === "authenticated" ? auth.username : null;

  React.useEffect(() => {
    if (open) setName("");
  }, [open]);

  const trimmed = name.trim();
  // Mirrors DatasetPicker: an exact repo_id match (case-insensitive) already
  // exists. The bare name may or may not carry a namespace, so also compare
  // against the trailing segment of each repo id.
  const matchesExisting = existingRepoIds.some((id) => {
    const bare = id.split("/").pop() ?? id;
    return (
      id.toLowerCase() === trimmed.toLowerCase() ||
      bare.toLowerCase() === trimmed.toLowerCase()
    );
  });
  const nameError = trimmed === "" ? null : validateDatasetName(trimmed);
  const canCreate = trimmed !== "" && nameError === null && !matchesExisting;

  // Live preview of the resulting repo id. Only shown when we know the
  // namespace and the user hasn't already typed their own "namespace/name"
  // (a "/" means they're supplying the namespace themselves — don't double
  // it up). Falls back to the placeholder while the field is empty.
  const showNamespaceHint = username !== null && !trimmed.includes("/");
  // Sample id, not prose: it is also the input's placeholder, and a dataset
  // name has to stay a valid ASCII repo segment. Never translated.
  const previewName = trimmed === "" ? SAMPLE_NAME : trimmed;

  const handleConfirm = () => {
    if (!canCreate) return;
    onCreateNew(trimmed);
    onOpenChange(false);
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>{t("landing.createDataset.title")}</DialogTitle>
          <DialogDescription className="text-muted-foreground">
            {t("landing.createDataset.description")}
          </DialogDescription>
        </DialogHeader>
        <form
          onSubmit={(e) => {
            e.preventDefault();
            handleConfirm();
          }}
          className="space-y-4"
        >
          <div>
            <Label htmlFor="new-dataset-name" className="text-muted-foreground">
              {t("landing.createDataset.nameLabel")}
            </Label>
            <Input
              id="new-dataset-name"
              autoFocus
              value={name}
              onChange={(e) =>
                setName(e.target.value.replace(/[^A-Za-z0-9._\-/]/g, "_"))
              }
              placeholder={SAMPLE_NAME}
              aria-invalid={nameError !== null || matchesExisting}
              className="mt-1 aria-[invalid=true]:border-destructive/70"
            />
            {matchesExisting ? (
              <p className="mt-1 text-xs text-destructive">
                {t("landing.createDataset.duplicate")}
              </p>
            ) : nameError ? (
              <p className="mt-1 text-xs text-destructive">{nameError}</p>
            ) : (
              showNamespaceHint && (
                <p className="mt-1 text-xs text-muted-foreground">
                  {/* The repo id is DATA — interpolated whole so no translation
                      ever splits a namespace from its name. */}
                  <Trans
                    i18nKey="landing.createDataset.creates"
                    values={{ repoId: `${username}/${previewName}` }}
                    components={[
                      <span key="0" className="font-mono text-foreground" />,
                    ]}
                  />
                </p>
              )
            )}
          </div>
          <DialogFooter>
            <Button
              type="button"
              variant="outline"
              onClick={() => onOpenChange(false)}
              className=""
            >
              {t("common.cancel")}
            </Button>
            <Button
              type="submit"
              disabled={!canCreate}
              className=""
            >
              <Plus className="w-4 h-4 mr-2" />{" "}
              {t("landing.createDataset.submit")}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
};

export default CreateDatasetDialog;
