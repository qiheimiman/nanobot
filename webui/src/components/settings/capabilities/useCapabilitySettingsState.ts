import { useState } from "react";

import {
  DEFAULT_IMAGE_GENERATION_FORM,
  imageGenerationFormFromPayload,
} from "@/components/settings/capabilities/ImageGenerationSettings";
import {
  DEFAULT_IMAGE_UNDERSTANDING_FORM,
  imageUnderstandingFormFromPayload,
} from "@/components/settings/capabilities/ImageUnderstandingSettings";
import {
  DEFAULT_NETWORK_SAFETY_FORM,
  networkSafetyFormFromPayload,
} from "@/components/settings/capabilities/SecuritySettings";
import {
  DEFAULT_TRANSCRIPTION_FORM,
  transcriptionFormFromPayload,
} from "@/components/settings/capabilities/TranscriptionSettings";
import {
  DEFAULT_WEB_SEARCH_FORM,
  webSearchFormFromPayload,
} from "@/components/settings/capabilities/WebSettings";
import type {
  ImageGenerationSettingsUpdate,
  ImageUnderstandingSettingsUpdate,
  NetworkSafetySettingsUpdate,
  SettingsPayload,
  TranscriptionSettingsUpdate,
  WebSearchSettingsUpdate,
} from "@/lib/types";

export type CapabilityErrorSection = "image" | "imageUnderstanding" | "voice" | "web" | "safety";

export function useCapabilitySettingsState(initialSettings: SettingsPayload | null) {
  const [capabilityErrors, setCapabilityErrors] = useState<Partial<Record<CapabilityErrorSection, string>>>({});
  const [webSearchSaving, setWebSearchSaving] = useState(false);
  const [imageGenerationSaving, setImageGenerationSaving] = useState(false);
  const [imageUnderstandingSaving, setImageUnderstandingSaving] = useState(false);
  const [transcriptionSaving, setTranscriptionSaving] = useState(false);
  const [networkSafetySaving, setNetworkSafetySaving] = useState(false);
  const [webSearchForm, setWebSearchForm] = useState<WebSearchSettingsUpdate>(() =>
    initialSettings ? webSearchFormFromPayload(initialSettings) : DEFAULT_WEB_SEARCH_FORM,
  );
  const [imageGenerationForm, setImageGenerationForm] = useState<ImageGenerationSettingsUpdate>(
    () => initialSettings
      ? imageGenerationFormFromPayload(initialSettings)
      : DEFAULT_IMAGE_GENERATION_FORM,
  );
  const [imageUnderstandingForm, setImageUnderstandingForm] = useState<ImageUnderstandingSettingsUpdate>(
    () => initialSettings
      ? imageUnderstandingFormFromPayload(initialSettings)
      : DEFAULT_IMAGE_UNDERSTANDING_FORM,
  );
  const [transcriptionForm, setTranscriptionForm] = useState<TranscriptionSettingsUpdate>(
    () => initialSettings ? transcriptionFormFromPayload(initialSettings) : DEFAULT_TRANSCRIPTION_FORM,
  );
  const [networkSafetyForm, setNetworkSafetyForm] = useState<NetworkSafetySettingsUpdate>(() =>
    initialSettings ? networkSafetyFormFromPayload(initialSettings) : DEFAULT_NETWORK_SAFETY_FORM,
  );
  const [webSearchKeyVisible, setWebSearchKeyVisible] = useState(false);
  const [webSearchKeyEditing, setWebSearchKeyEditing] = useState(false);

  return {
    capabilityErrors,
    setCapabilityErrors,
    imageGenerationForm,
    imageGenerationSaving,
    imageUnderstandingForm,
    imageUnderstandingSaving,
    networkSafetyForm,
    networkSafetySaving,
    setImageGenerationForm,
    setImageGenerationSaving,
    setImageUnderstandingForm,
    setImageUnderstandingSaving,
    setNetworkSafetyForm,
    setNetworkSafetySaving,
    setTranscriptionForm,
    setTranscriptionSaving,
    setWebSearchForm,
    setWebSearchKeyEditing,
    setWebSearchKeyVisible,
    setWebSearchSaving,
    transcriptionForm,
    transcriptionSaving,
    webSearchForm,
    webSearchKeyEditing,
    webSearchKeyVisible,
    webSearchSaving,
  };
}

export type CapabilitySettingsState = ReturnType<typeof useCapabilitySettingsState>;
