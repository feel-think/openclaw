// PATCH: footer — delegate to footer-enhanced.mjs for enhanced usage footer
import { createRequire } from "node:module";
import { estimateUsageCost, formatTokenCount, formatUsd } from "../../utils/usage-format.js";
import type { ReplyPayload } from "../types.js";

const _require = createRequire(import.meta.url);
const FOOTER_PATH =
  "/home/ubuntu/.openclaw/workspace/projects/openclaw-patches/footer/footer-enhanced.mjs";
let _genFooter: ((p: Record<string, unknown>) => string | null) | null = null;

function _loadFooter(): typeof _genFooter {
  if (_genFooter !== null) {
    return _genFooter;
  }
  try {
    const mod = _require(FOOTER_PATH);
    _genFooter = mod.generateFooter ?? null;
  } catch {
    _genFooter = null;
  }
  return _genFooter;
}

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
  /** PATCH: footer — extra params for enhanced footer */
  contextTokens?: number;
  lastCallUsage?: Record<string, unknown>;
  sessionKey?: string;
  systemPromptReport?: Record<string, unknown>;
}): string | null => {
  // Try enhanced footer first
  const gen = _loadFooter();
  if (gen) {
    try {
      const result = gen({
        ...params,
        formatTokenCount,
        estimateUsageCost,
        formatUsd,
      });
      if (result) {
        return result;
      }
    } catch {
      // Fall through to default
    }
  }

  // Default footer (fallback if footer-enhanced.mjs unavailable)
  const usage = params.usage;
  if (!usage) {
    return null;
  }
  const input = usage.input;
  const output = usage.output;
  if (typeof input !== "number" && typeof output !== "number") {
    return null;
  }
  const inputLabel = typeof input === "number" ? formatTokenCount(input) : "?";
  const outputLabel = typeof output === "number" ? formatTokenCount(output) : "?";
  const cost =
    params.showCost && typeof input === "number" && typeof output === "number"
      ? estimateUsageCost({
          usage: {
            input,
            output,
            cacheRead: usage.cacheRead,
            cacheWrite: usage.cacheWrite,
          },
          cost: params.costConfig,
        })
      : undefined;
  const costLabel = params.showCost ? formatUsd(cost) : undefined;
  const suffix = costLabel ? ` · est ${costLabel}` : "";
  return `Usage: ${inputLabel} in / ${outputLabel} out${suffix}`;
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
