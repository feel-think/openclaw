import { estimateUsageCost, formatTokenCount, formatUsd } from "../../utils/usage-format.js";
import type { ReplyPayload } from "../types.js";
import { createRequire } from "node:module";
import { homedir } from "node:os";

const _require = createRequire(import.meta.url);
const _FOOTER_PATH = homedir() + "/.openclaw/workspace/projects/openclaw-patches/footer/footer-enhanced.mjs";

export const formatResponseUsageLine = (params: {
  usage?: {
    input?: number;
    output?: number;
    cacheRead?: number;
    cacheWrite?: number;
  };
  showCost: boolean;
  costConfig?: {
    input: number;
    output: number;
    cacheRead: number;
    cacheWrite: number;
  };
  contextTokens?: number;
  lastCallUsage?: {
    input?: number;
    output?: number;
    cacheRead?: number;
    cacheWrite?: number;
    total?: number;
  };
  sessionKey?: string;
  systemPromptReport?: Record<string, unknown>;
}): string | null => {
  const usage = params.usage;
  if (!usage) {
    return null;
  }
  try {
    const { generateFooter } = _require(_FOOTER_PATH);
    return generateFooter({
      ...params,
      formatTokenCount,
      estimateUsageCost,
      formatUsd,
    });
  } catch {
    // Fallback to original logic if footer-enhanced.mjs not available
    const input = usage.input;
    const output = usage.output;
    if (typeof input !== "number" && typeof output !== "number") {
      return null;
    }
    const inputLabel = typeof input === "number" ? formatTokenCount(input) : "?";
    const outputLabel = typeof output === "number" ? formatTokenCount(output) : "?";
    const cacheRead = typeof usage.cacheRead === "number" ? usage.cacheRead : undefined;
    const cacheWrite = typeof usage.cacheWrite === "number" ? usage.cacheWrite : undefined;
    const cost =
      params.showCost && typeof input === "number" && typeof output === "number"
        ? estimateUsageCost({
            usage: { input, output, cacheRead: usage.cacheRead, cacheWrite: usage.cacheWrite },
            cost: params.costConfig,
          })
        : undefined;
    const costLabel = params.showCost ? formatUsd(cost) : undefined;
    const cacheSuffix =
      (typeof cacheRead === "number" && cacheRead > 0) ||
      (typeof cacheWrite === "number" && cacheWrite > 0)
        ? ` · cache ${formatTokenCount(cacheRead ?? 0)} cached / ${formatTokenCount(cacheWrite ?? 0)} new`
        : "";
    const suffix = costLabel ? ` · est ${costLabel}` : "";
    return `Usage: ${inputLabel} in / ${outputLabel} out${cacheSuffix}${suffix}`;
  }
};

export const appendUsageLine = (payloads: ReplyPayload[], line: string): ReplyPayload[] => {
  let index = -1;
  for (let i = payloads.length - 1; i >= 0; i -= 1) {
    if (payloads[i]?.text) {
      index = i;
      break;
    }
  }
  if (index === -1) {
    return [...payloads, { text: line }];
  }
  const existing = payloads[index];
  const existingText = existing.text ?? "";
  const separator = existingText.endsWith("\n") ? "" : "\n";
  const next = {
    ...existing,
    text: `${existingText}${separator}${line}`,
  };
  const updated = payloads.slice();
  updated[index] = next;
  return updated;
};
