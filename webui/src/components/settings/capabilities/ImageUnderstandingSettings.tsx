import { useAutoSave } from "@/components/settings/shared/useAutoSave";
import type { Dispatch, SetStateAction } from "react";
import { useTranslation } from "react-i18next";

import { ModelIdPicker, ProviderPicker } from "@/components/settings/shared/ModelControls";
import {
  RestartSettingsFooter,
  SettingsGroup,
  SettingsRow,
  SettingsSectionTitle,
  StatusPill,
} from "@/components/settings/shared/SettingsControls";
import { ToggleButton } from "@/components/settings/ToggleButton";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import type { ImageUnderstandingSettingsUpdate, SettingsPayload } from "@/lib/types";

export const DEFAULT_IMAGE_UNDERSTANDING_FORM: ImageUnderstandingSettingsUpdate = {
  enabled: false,
  provider: "deepseek",
  model: "deepseek-flash",
  prompt: "",
};

export function imageUnderstandingFormFromPayload(payload: SettingsPayload): ImageUnderstandingSettingsUpdate {
  const iu = payload.image_understanding;
  return {
    enabled: iu?.enabled ?? false,
    provider: iu?.provider ?? "deepseek",
    model: iu?.model ?? "deepseek-flash",
    prompt: iu?.prompt ?? "",
  };
}

export function ImageUnderstandingSettings({
  embedded = false,
  error,
  children,
  token,
  settings,
  form,
  dirty,
  saving,
  onChangeForm,
  onSave,
  onOpenProviders,
  showBrandLogos,
  onRestart,
  isRestarting,
  requiresRestartPending,
}: {
  embedded?: boolean;
  error?: string;
  children?: React.ReactNode;
  token: string;
  settings: SettingsPayload;
  form: ImageUnderstandingSettingsUpdate;
  dirty: boolean;
  saving: boolean;
  onChangeForm: Dispatch<SetStateAction<ImageUnderstandingSettingsUpdate>>;
  onSave: () => void;
  onOpenProviders: () => void;
  showBrandLogos: boolean;
  onRestart?: () => void;
  isRestarting?: boolean;
  requiresRestartPending: boolean;
}) {
  const { t } = useTranslation();
  const tx = (key: string, fallback: string) => t(key, { defaultValue: fallback });
  const selectedProvider =
    settings.image_understanding?.providers.find((provider) => provider.name === form.provider) ??
    settings.image_understanding?.providers[0];
  const providerConfigured = !!selectedProvider?.configured;
  const missingCredential = form.enabled && !providerConfigured;
  useAutoSave(form, dirty, saving, onSave, !embedded && !missingCredential);

  return (
    <div className="settings-stack">
      <section>
        {!embedded ? <SettingsSectionTitle>{tx("settings.sections.imageUnderstanding", "Image understanding")}</SettingsSectionTitle> : null}
        <SettingsGroup>
          {!embedded ? (
          <SettingsRow title={tx("settings.rows.imageUnderstanding", "Image understanding")}>
            <ToggleButton
              checked={form.enabled}
              onChange={(enabled) => onChangeForm((prev) => ({ ...prev, enabled }))}
              ariaLabel={tx("settings.rows.imageUnderstanding", "Image understanding")}
              label={form.enabled ? tx("settings.values.on", "On") : tx("settings.values.off", "Off")}
            />
          </SettingsRow>
          ) : null}
          {embedded || form.enabled ? <>
          <SettingsRow title={tx("settings.rows.imageProvider", "Provider")}>
            <ProviderPicker
              providers={settings.image_understanding?.providers ?? []}
              value={form.provider}
              emptyLabel={tx("settings.imageUnderstanding.selectProvider", "Select provider")}
              showProviderLogos={showBrandLogos}
              onChange={(provider) => {
                const nextProvider = settings.image_understanding?.providers.find((row) => row.name === provider);
                onChangeForm((prev) => ({
                  ...prev,
                  provider,
                  model: nextProvider?.models?.[0] || prev.model,
                }));
              }}
            />
          </SettingsRow>
          <SettingsRow
            title={tx("settings.rows.imageProviderStatus", "Provider status")}
            description={tx("settings.help.imageUnderstandingProviderStatus", "Reuses provider credentials from Models settings.")}
          >
            <div className="flex flex-wrap items-center justify-end gap-2">
              <StatusPill tone={providerConfigured ? "success" : "neutral"}>
                {providerConfigured
                  ? tx("settings.values.configured", "Configured")
                  : tx("settings.values.notConfigured", "Not configured")}
              </StatusPill>
              {!providerConfigured ? (
                <Button size="sm" variant="outline" onClick={onOpenProviders} className="rounded-full">
                  {tx("settings.image.configureProvider", "Configure provider")}
                </Button>
              ) : null}
            </div>
          </SettingsRow>
          <SettingsRow title={tx("settings.rows.imageProviderBase", "Provider URL")}>
            <span className="max-w-[320px] truncate text-right text-[13px] text-muted-foreground">
              {selectedProvider?.api_base || selectedProvider?.default_api_base || tx("settings.values.notAvailable", "Not available")}
            </span>
          </SettingsRow>
          <SettingsRow title={tx("settings.rows.imageModel", "Vision model")}>
            <ModelIdPicker
              token={token}
              settings={settings}
              provider={form.provider}
              models={selectedProvider?.models ?? []}
              value={form.model}
              showProviderLogos={showBrandLogos}
              emptyLabel={tx("settings.imageUnderstanding.selectModel", "Select vision model")}
              searchPlaceholder={tx(
                "settings.image.searchOrTypeModel",
                "Search or type model ID",
              )}
              emptyMessage={tx(
                "settings.image.typeModelId",
                "Type the model ID supported by this provider.",
              )}
              onChange={(model) => onChangeForm((prev) => ({ ...prev, model }))}
            />
          </SettingsRow>
          <SettingsRow
            title={tx("settings.rows.imageUnderstandingPrompt", "Analysis prompt")}
            description={tx("settings.help.imageUnderstandingPrompt", "Prompt sent to the vision model. Defaults to '描述这张图片' if empty.")}
          >
            <Input
              className="max-w-[320px]"
              value={form.prompt}
              onChange={(e) => onChangeForm((prev) => ({ ...prev, prompt: e.target.value }))}
              placeholder={tx("settings.imageUnderstanding.promptPlaceholder", "描述这张图片")}
            />
          </SettingsRow>
          {children}
          </> : null}
          <RestartSettingsFooter
            error={missingCredential || Boolean(error)}
            autoSave
            dirty={dirty}
            saving={saving}
            pendingRestart={!embedded && requiresRestartPending}
            disabled={missingCredential}
            message={
              missingCredential
                ? tx("settings.imageUnderstanding.missingCredential", "Configure this provider before enabling image understanding.")
                : error
            }
            dirtyMessage={tx("settings.status.restartAfterSaving", "Save changes, then restart nanobot to apply them.")}
            pendingMessage={tx("settings.status.savedRestartApply", "Saved. Restart to apply changes.")}
            onSave={onSave}
            onRestart={onRestart}
            isRestarting={isRestarting}
          />
        </SettingsGroup>
      </section>
    </div>
  );
}
