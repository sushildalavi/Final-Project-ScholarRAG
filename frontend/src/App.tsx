import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { API_BASE, api } from './api/client';
import {
  Citation,
  ConfidenceObject,
  DocumentRow,
  EvalCase,
  EvalRunResponse,
  FaithfulnessReport,
  JudgeCasePayload, JudgeRunResponse, JudgeRunSummary,
  MsaCalibrationPayload, MsaCalibrationResponse, MsaCalibrationLatest,
  WhyTraceChunk,
} from './api/types';
import {
  DEFAULT_EVAL_PRESET_COUNT,
  buildJudgePresetCases,
  buildRetrievalPresetCases,
  selectPresetDocuments,
} from './lib/evalPresets';
import { PromptInputBox } from '@/components/ui/ai-prompt-box';
import { MeshGradient } from '@paper-design/shaders-react';
import { motion, AnimatePresence } from 'framer-motion';

// ── Types ─────────────────────────────────────────────────────────────────────
type Page = 'studio' | 'eval';

type UiMessage = {
  role: 'you' | 'assistant';
  text: string;
  streaming?: boolean;
  citations?: Citation[];
  confidence?: ConfidenceObject;
  why_answer?: { rerank_changed_order: boolean; top_chunks: WhyTraceChunk[] };
  latency_breakdown_ms?: { retrieve: number; rerank: number; generate: number; total: number };
  needs_clarification?: boolean;
  clarification?: { question: string; options: string[]; recommended_option?: string } | null;
  faithfulness?: FaithfulnessReport | null;
  answer_scope?: string;
  unsupported_claims?: number;
  query_ref?: string;
  retrieval_policy?: {
    public_provider_status?: Record<
      string,
      {
        selected?: number;
        fetched?: number;
        contributed?: boolean;
      }
    >;
  };
};

type SourceRow = {
  id: number;
  title: string;
  doc_id?: number;
  msa?: { msa_score: number };
  msa_supported?: boolean;
  url?: string;
  source?: string;
  page?: number;
  pages?: number[];
  cited?: boolean;
  citation_count?: number;
  excerpt_count?: number;
  snippet_preview?: string;
  confidence_obj?: ConfidenceObject;
};

type EvidenceState = {
  citations: Citation[];
  trace: WhyTraceChunk[];
};

function sourceDedupKey(citation: Pick<Citation, 'doc_id' | 'source' | 'url' | 'title'>): string {
  return citation.doc_id
    ? `uploaded|${citation.doc_id}`
    : `${citation.source || ''}|${citation.url || ''}|${citation.title || ''}`;
}

function uniqueSourceCount(citations?: Citation[]): number {
  const keys = new Set<string>();
  for (const citation of citations || []) {
    keys.add(sourceDedupKey(citation));
  }
  return keys.size;
}

type StudioSession = {
  id: string;
  title: string;
  messages: UiMessage[];
  selectedDocs: number[];
  activeEvidence: EvidenceState;
  activeEvidenceMsgIdx: number;
  allowGeneralBackground: boolean;
  createdAt: number;
  updatedAt: number;
};

// ── Markdown renderer ─────────────────────────────────────────────────────────
function renderInline(text: string): ReactNode {
  const nodes: ReactNode[] = [];
  const pattern = /(\*\*(.+?)\*\*|\*(.+?)\*|`([^`]+)`|\[S?(\d+)\])/g;
  let last = 0;
  let m: RegExpExecArray | null;
  while ((m = pattern.exec(text)) !== null) {
    if (m.index > last) nodes.push(text.slice(last, m.index));
    if (m[2] != null)      nodes.push(<strong key={m.index}>{m[2]}</strong>);
    else if (m[3] != null) nodes.push(<em key={m.index}>{m[3]}</em>);
    else if (m[4] != null) nodes.push(<code key={m.index}>{m[4]}</code>);
    else if (m[5] != null) nodes.push(<span key={m.index} className="cite-chip">[S{m[5]}]</span>);
    last = m.index + m[0].length;
  }
  if (last < text.length) nodes.push(text.slice(last));
  return nodes.length === 1 ? nodes[0] : <>{nodes}</>;
}

function renderMarkdown(raw: string): ReactNode {
  const lines = (raw || '').split('\n');
  const out: ReactNode[] = [];
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    const trimmed = line.trim();
    if (!trimmed) { i++; continue; }

    if (/^[-*_]{3,}$/.test(trimmed)) { out.push(<hr key={i} />); i++; continue; }

    const hm = line.match(/^(#{1,6})\s+(.+)/);
    if (hm) {
      const lvl = Math.min(hm[1].length, 3);
      const Tag = `h${lvl}` as 'h1' | 'h2' | 'h3';
      out.push(<Tag key={i}>{renderInline(hm[2])}</Tag>);
      i++; continue;
    }

    if (line.startsWith('> ')) {
      out.push(<blockquote key={i}>{renderInline(line.slice(2))}</blockquote>);
      i++; continue;
    }

    if (/^[-*+]\s/.test(line)) {
      const items: ReactNode[] = [];
      while (i < lines.length && /^[-*+]\s/.test(lines[i])) {
        items.push(<li key={i}>{renderInline(lines[i].replace(/^[-*+]\s+/, ''))}</li>);
        i++;
      }
      out.push(<ul key={`ul${i}`}>{items}</ul>);
      continue;
    }

    if (/^\d+[.)]\s/.test(line)) {
      const items: ReactNode[] = [];
      while (i < lines.length && /^\d+[.)]\s/.test(lines[i])) {
        items.push(<li key={i}>{renderInline(lines[i].replace(/^\d+[.)]\s+/, ''))}</li>);
        i++;
      }
      out.push(<ol key={`ol${i}`}>{items}</ol>);
      continue;
    }

    const pLines: string[] = [];
    while (i < lines.length) {
      const l = lines[i];
      const t = l.trim();
      if (!t || /^[-*_]{3,}$/.test(t) || /^#{1,6}\s/.test(l) ||
          l.startsWith('> ') || /^[-*+]\s/.test(l) || /^\d+[.)]\s/.test(l)) break;
      pLines.push(l);
      i++;
    }
    if (pLines.length) out.push(<p key={`p${i}`}>{renderInline(pLines.join(' '))}</p>);
  }
  return <div className="md">{out}</div>;
}

function SettingsIcon() {
  return (
    <svg viewBox="0 0 24 24" className="workspace-inline-icon" aria-hidden="true" fill="none">
      <path d="M10.3 3.4a1 1 0 0 1 1.4-.5l.3.2a2 2 0 0 0 2 0l.3-.2a1 1 0 0 1 1.4.5l.5 1a2 2 0 0 0 1.5 1.1l1.1.2a1 1 0 0 1 .8 1.3l-.1.3a2 2 0 0 0 .4 2l.7.9a1 1 0 0 1 0 1.4l-.7.9a2 2 0 0 0-.4 2l.1.3a1 1 0 0 1-.8 1.3l-1.1.2a2 2 0 0 0-1.5 1.1l-.5 1a1 1 0 0 1-1.4.5l-.3-.2a2 2 0 0 0-2 0l-.3.2a1 1 0 0 1-1.4-.5l-.5-1a2 2 0 0 0-1.5-1.1l-1.1-.2a1 1 0 0 1-.8-1.3l.1-.3a2 2 0 0 0-.4-2l-.7-.9a1 1 0 0 1 0-1.4l.7-.9a2 2 0 0 0 .4-2l-.1-.3a1 1 0 0 1 .8-1.3l1.1-.2A2 2 0 0 0 9.8 5l.5-1Z" stroke="currentColor" strokeWidth="1.5" />
      <circle cx="12" cy="12" r="3" stroke="currentColor" strokeWidth="1.5" />
    </svg>
  );
}

// ── Confidence badge ──────────────────────────────────────────────────────────
function confidenceTooltip(c: ConfidenceObject): string {
  const f = c.factors;
  const msa = f.msa ? ` | M=${f.msa.M.toFixed(2)} S=${f.msa.S.toFixed(2)} A=${f.msa.A.toFixed(2)}` : '';
  return `sim=${f.top_sim.toFixed(3)} cov=${f.citation_coverage.toFixed(3)} margin=${f.evidence_margin.toFixed(3)}${msa}`;
}

function ConfBadge({
  confidence,
  showWhenMissing = false,
}: {
  confidence?: ConfidenceObject;
  showWhenMissing?: boolean;
}) {
  if (!confidence) {
    return showWhenMissing ? <span className="conf-badge unknown">Confidence: N/A</span> : null;
  }
  if (confidence.needs_clarification) {
    return <span className="conf-badge needs-clarification">Confidence: Clarify</span>;
  }
  const pct = Math.round((confidence.score || 0) * 100);
  const raw = (confidence.label || 'Low').toLowerCase();
  const cls = ['high', 'med', 'low'].includes(raw) ? raw : 'low';
  return (
    <span className={`conf-badge ${cls}`} title={confidenceTooltip(confidence)}>
      Confidence: {confidence.label} {pct}%
    </span>
  );
}

type InlineCitationItem = {
  id: number;
  title: string;
  page?: number;
  usedInAnswer: boolean;
};

function InlineCitations({
  citations,
  onActivateEvidence,
}: {
  citations?: Citation[];
  onActivateEvidence: () => void;
}) {
  const items = useMemo<InlineCitationItem[]>(() => {
    const seen = new Set<string>();
    const out: InlineCitationItem[] = [];
    for (let i = 0; i < (citations || []).length; i++) {
      const citation = citations?.[i];
      if (!citation) continue;
      const id = typeof citation.id === 'number' ? citation.id : i + 1;
      const key = `${id}|${sourceDedupKey(citation)}`;
      if (seen.has(key)) continue;
      seen.add(key);
      out.push({
        id,
        title: citation.title || `Document ${citation.doc_id ?? '?'}`,
        page: typeof citation.page === 'number' ? citation.page : undefined,
        usedInAnswer: Boolean(citation.used_in_answer),
      });
    }
    return out
      .sort((a, b) => Number(b.usedInAnswer) - Number(a.usedInAnswer) || a.id - b.id)
      .slice(0, 8);
  }, [citations]);

  if (!items.length) return null;
  return (
    <div className="inline-citations">
      <span className="inline-citations-label">Citations</span>
      {items.map((item) => (
        <button
          key={`${item.id}-${item.title}-${item.page ?? 'na'}`}
          type="button"
          className={`inline-citation${item.usedInAnswer ? ' used' : ''}`}
          onClick={onActivateEvidence}
          title={`${item.title}${item.page != null ? ` · p.${item.page}` : ''}`}
        >
          <span className="inline-citation-tag">[S{item.id}]</span>
          <span className="inline-citation-text">{item.title}</span>
          {item.page != null ? <span className="inline-citation-page">p.{item.page}</span> : null}
        </button>
      ))}
    </div>
  );
}

function formatAnswerScope(scope?: string): string | null {
  if (!scope) return null;
  const normalized = scope.trim().toLowerCase();
  const explicit: Record<string, string> = {
    official_document_context: 'Official document',
    uploaded_document_context: 'Uploaded document',
    personal_document_context: 'Personal document',
    public_research_context: 'Public research',
    mixed_research_context: 'Mixed research',
    context_limited: 'Context limited',
  };
  if (explicit[normalized]) return explicit[normalized];
  return normalized
    .replace(/_context$/, '')
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

function formatProviderSummary(message: UiMessage): string | null {
  const status = message.retrieval_policy?.public_provider_status;
  if (!status) return null;
  const parts = Object.entries(status)
    .filter(([, meta]) => (meta?.selected || 0) > 0)
    .sort((a, b) => (b[1]?.selected || 0) - (a[1]?.selected || 0))
    .map(([provider, meta]) => `${provider}:${meta?.selected || 0}/${meta?.fetched || 0}`);
  if (!parts.length) return null;
  return `Providers ${parts.join(' · ')}`;
}

function fmtPct(value: number | undefined | null, digits = 1): string {
  if (value == null || Number.isNaN(value)) return '—';
  return `${(value * 100).toFixed(digits)}%`;
}

function fmtNum(value: number | undefined | null, digits = 3): string {
  if (value == null || Number.isNaN(value)) return '—';
  return Number(value).toFixed(digits);
}

function ConfidenceDiagnostics({ message }: { message: UiMessage }) {
  const confidence = message.confidence;
  const factors = confidence?.factors;
  const msa = factors?.msa;
  const faithfulness = message.faithfulness;
  const citations = message.citations || [];
  const citedCount = citations.filter((c) => c.used_in_answer).length;
  const totalCitations = citations.length;
  const providerSummary = formatProviderSummary(message);

  return (
    <div className="confidence-diagnostics">
      <div className="confidence-diagnostics-head">
        <span className="confidence-diagnostics-title">Confidence diagnostics</span>
        <span className="confidence-diagnostics-score">
          {confidence ? `${confidence.label} ${fmtPct(confidence.score, 0)}` : 'N/A'}
        </span>
      </div>

      <div className="confidence-diagnostics-grid">
        <div className="confidence-diagnostics-card">
          <div className="confidence-diagnostics-card-title">Evidence usage</div>
          <div className="confidence-diagnostics-row"><span>Cited chunks</span><strong>{citedCount}/{totalCitations || 0}</strong></div>
          <div className="confidence-diagnostics-row"><span>Unique sources</span><strong>{uniqueSourceCount(citations)}</strong></div>
          <div className="confidence-diagnostics-row"><span>Coverage</span><strong>{fmtPct(factors?.citation_coverage, 1)}</strong></div>
          {providerSummary ? <div className="confidence-diagnostics-note">{providerSummary}</div> : null}
        </div>

        <div className="confidence-diagnostics-card">
          <div className="confidence-diagnostics-card-title">Retrieval quality</div>
          <div className="confidence-diagnostics-row"><span>Top similarity</span><strong>{fmtNum(factors?.top_sim, 3)}</strong></div>
          <div className="confidence-diagnostics-row"><span>Top rerank norm</span><strong>{fmtNum(factors?.top_rerank_norm, 3)}</strong></div>
          <div className="confidence-diagnostics-row"><span>Evidence margin</span><strong>{fmtNum(factors?.evidence_margin, 3)}</strong></div>
          {typeof factors?.minimum_score === 'number' ? (
            <div className="confidence-diagnostics-row"><span>Grounded floor</span><strong>{fmtPct(factors.minimum_score, 0)}</strong></div>
          ) : null}
        </div>

        <div className="confidence-diagnostics-card">
          <div className="confidence-diagnostics-card-title">Penalties</div>
          <div className="confidence-diagnostics-row"><span>Ambiguity</span><strong>{fmtPct(factors?.ambiguity_penalty, 1)}</strong></div>
          <div className="confidence-diagnostics-row"><span>Insufficiency</span><strong>{fmtPct(factors?.insufficiency_penalty, 1)}</strong></div>
          <div className="confidence-diagnostics-row"><span>Scope</span><strong>{fmtPct(factors?.scope_penalty, 1)}</strong></div>
          <div className="confidence-diagnostics-row"><span>Needs clarification</span><strong>{confidence?.needs_clarification ? 'Yes' : 'No'}</strong></div>
        </div>

        <div className="confidence-diagnostics-card">
          <div className="confidence-diagnostics-card-title">M/S/A + faithfulness</div>
          <div className="confidence-diagnostics-row"><span>M</span><strong>{fmtNum(msa?.M, 3)}</strong></div>
          <div className="confidence-diagnostics-row"><span>S</span><strong>{fmtNum(msa?.S, 3)}</strong></div>
          <div className="confidence-diagnostics-row"><span>A</span><strong>{fmtNum(msa?.A, 3)}</strong></div>
          <div className="confidence-diagnostics-row"><span>MSA score</span><strong>{fmtPct(msa?.msa_score, 1)}</strong></div>
          <div className="confidence-diagnostics-row"><span>Faithfulness</span><strong>{fmtPct(faithfulness?.overall_score, 0)}</strong></div>
          {faithfulness ? (
            <div className="confidence-diagnostics-row"><span>Unsupported claims</span><strong>{faithfulness.unsupported_count ?? 0}</strong></div>
          ) : (
            <div className="confidence-diagnostics-note">Faithfulness judge was not run for this answer.</div>
          )}
        </div>
      </div>

      {confidence?.explanation ? (
        <div className="confidence-diagnostics-note">{confidence.explanation}</div>
      ) : null}
    </div>
  );
}

// ── Source card ───────────────────────────────────────────────────────────────
function SourceCard({ row, idx, onClick }: { row: SourceRow; idx: number; onClick: () => void }) {
  const msaScore = row.msa?.msa_score;
  const confPct = msaScore != null
    ? Math.round(msaScore * 100)
    : row.confidence_obj ? Math.round((row.confidence_obj.score || 0) * 100) : null;
  const rawLabel = msaScore != null
    ? (row.msa_supported ? 'high' : 'low')
    : (row.confidence_obj?.label?.toLowerCase() || 'default');
  const confCls = ['high', 'med', 'low'].includes(rawLabel) ? rawLabel : 'default';
  const pages = (row.pages || []).filter((p): p is number => typeof p === 'number');
  const pageLabel =
    pages.length > 1
      ? `pp.${pages.slice(0, 4).join(', ')}${pages.length > 4 ? '…' : ''}`
      : typeof row.page === 'number'
        ? `p.${row.page}`
        : null;

  return (
    <motion.button
      className={`source-card${row.cited ? ' cited' : ''}`}
      onClick={onClick}
      initial={{ opacity: 0, y: 14 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, delay: idx * 0.06 }}
      whileHover={{ y: -3, scale: 1.01 }}
    >
      <div className="sc-overline">
        <span className="sc-source-tag">S{row.id}</span>
        <span className="sc-meta-tag">{String(row.source || 'source')}</span>
        {pageLabel && <span className="sc-meta-tag">{pageLabel}</span>}
        {(row.excerpt_count || 0) > 1 && <span className="sc-meta-tag">{row.excerpt_count} excerpts</span>}
      </div>
      <div className="sc-head">
        <span className="sc-title">{row.title || `Document ${row.doc_id ?? '?'}`}</span>
        {confPct != null && <span className={`sc-conf ${confCls}`}>{confPct}%</span>}
      </div>
      <div className="sc-foot">
        <span className={`sc-foot-state${row.cited ? ' cited' : ''}`}>{row.cited ? 'Cited' : 'Retrieved'}</span>
        {row.url && (
          <a className="sc-link" href={row.url} target="_blank" rel="noreferrer"
             onClick={(e) => e.stopPropagation()}>Open ↗</a>
        )}
      </div>
      {row.snippet_preview && <div className="sc-snippet">{row.snippet_preview}</div>}
    </motion.button>
  );
}

// ── Typing indicator ──────────────────────────────────────────────────────────
function TypingIndicator() {
  return (
    <motion.div
      className="typing-row"
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -8 }}
      transition={{ duration: 0.25 }}
    >
      <div className="msg-avatar assistant" aria-hidden="true">
        <span className="sigil" />
      </div>
      <div className="typing-bubble">
        {[0, 1, 2].map((i) => (
          <motion.div
            key={i}
            className="t-dot"
            animate={{ y: [0, -6, 0], opacity: [0.4, 1, 0.4] }}
            transition={{ duration: 1.2, repeat: Infinity, delay: i * 0.16 }}
          />
        ))}
      </div>
    </motion.div>
  );
}

// ── Source detail modal ───────────────────────────────────────────────────────
function SourceModal({ row, onClose }: { row: SourceRow | null; onClose: () => void }) {
  if (!row) return null;
  const pages = (row.pages || []).filter((p): p is number => typeof p === 'number');
  const pageLabel =
    pages.length > 1
      ? ` · pp.${pages.join(', ')}`
      : typeof row.page === 'number'
        ? ` · p.${row.page}`
        : '';
  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-card" onClick={(e) => e.stopPropagation()}>
        <div className="modal-head">
          <div>
            <div className="modal-head-title">{row.title || `Document ${row.doc_id ?? '?'}`}</div>
            <div className="modal-head-meta">
              S{row.id} · {String(row.source || 'source')}{pageLabel}
            </div>
          </div>
          <button className="modal-close" onClick={onClose}>✕</button>
        </div>
        <div className="modal-body">{row.snippet_preview || 'No snippet available.'}</div>
        {row.url && (
          <div className="modal-actions">
            <a className="btn btn-ghost btn-sm" href={row.url} target="_blank" rel="noreferrer">
              Open source ↗
            </a>
          </div>
        )}
      </div>
    </div>
  );
}

// ── Delete confirm modal ──────────────────────────────────────────────────────
function DeleteModal({
  doc, onCancel, onConfirm,
}: { doc: DocumentRow | null; onCancel: () => void; onConfirm: () => void }) {
  if (!doc) return null;
  return (
    <div className="modal-backdrop" onClick={onCancel}>
      <div className="modal-card" onClick={(e) => e.stopPropagation()}>
        <div className="modal-head">
          <div className="modal-head-title">Delete document</div>
          <button className="modal-close" onClick={onCancel}>✕</button>
        </div>
        <div className="modal-body">
          <p>Remove <strong>{doc.title}</strong> from your workspace?</p>
          <p style={{ marginTop: 10 }}>
            All chunks and embeddings will be deleted permanently.
          </p>
        </div>
        <div className="modal-actions">
          <button className="btn btn-ghost btn-sm" onClick={onCancel}>Cancel</button>
          <button className="btn btn-danger btn-sm" onClick={onConfirm}>Delete</button>
        </div>
      </div>
    </div>
  );
}

// ── Evidence panel ────────────────────────────────────────────────────────────
function EvidencePanel({
  citations, traceChunks, loading, allowGeneralBackground,
}: {
  citations: Citation[];
  traceChunks: WhyTraceChunk[];
  loading: boolean;
  allowGeneralBackground: boolean;
}) {
  const [showCitedOnly, setShowCitedOnly] = useState(true);
  const [modalRow, setModalRow] = useState<SourceRow | null>(null);

  const rows = useMemo<SourceRow[]>(() => {
    const traceById = new Map<number, WhyTraceChunk>();
    (traceChunks || []).forEach((t, i) => traceById.set(t.id || i + 1, t));
    const mapped = (citations || []).map((c, i) => {
      const id = c.id || i + 1;
      const trace = traceById.get(id);
      return {
        id,
        title: c.title || trace?.title || `Document ${c.doc_id ?? trace?.doc_id ?? '?'}`,
        doc_id: c.doc_id ?? trace?.doc_id,
        msa: c.msa,
        msa_supported: c.msa_supported,
        url: c.url,
        source: c.source,
        page: c.page ?? trace?.page,
        cited: c.used_in_answer ?? trace?.cited,
        snippet_preview: trace?.snippet_preview || '',
        confidence_obj: c.confidence_obj,
      };
    });
    const deduped = new Map<string, SourceRow>();
    for (const row of mapped) {
      const key = sourceDedupKey(row);
      const existing = deduped.get(key);
      if (!existing) {
        deduped.set(key, {
          ...row,
          pages: typeof row.page === 'number' ? [row.page] : [],
          citation_count: 1,
          excerpt_count: 1,
        });
        continue;
      }
      const mergedPages = Array.from(new Set([
        ...(existing.pages || (typeof existing.page === 'number' ? [existing.page] : [])),
        ...(typeof row.page === 'number' ? [row.page] : []),
      ])).sort((a, b) => a - b);
      deduped.set(key, {
        ...existing,
        cited: Boolean(existing.cited || row.cited),
        page: mergedPages[0] ?? existing.page ?? row.page,
        pages: mergedPages,
        citation_count: (existing.citation_count || 1) + 1,
        excerpt_count: (existing.excerpt_count || 1) + 1,
        snippet_preview: existing.snippet_preview || row.snippet_preview,
        confidence_obj:
          (row.confidence_obj?.score || 0) > (existing.confidence_obj?.score || 0)
            ? row.confidence_obj
            : existing.confidence_obj,
        msa:
          (row.msa?.msa_score || 0) > (existing.msa?.msa_score || 0)
            ? row.msa
            : existing.msa,
        msa_supported: Boolean(existing.msa_supported || row.msa_supported),
      });
    }
    return Array.from(deduped.values()).map((row, idx) => ({ ...row, id: idx + 1 }));
  }, [citations, traceChunks]);

  const visible = useMemo(() => {
    const filtered = showCitedOnly ? rows.filter((r) => r.cited) : rows;
    return filtered.length ? filtered : rows;
  }, [rows, showCitedOnly]);

  const citedCount = rows.filter((r) => r.cited).length;

  return (
    <>
      <div className="evidence-panel">
        <div className="evidence-head">
          <div className="evidence-head-row">
            <div>
              <div className="evidence-head-kicker">Inspector</div>
              <div className="evidence-head-title">Evidence</div>
              <div className="evidence-head-sub">
                {loading
                  ? (allowGeneralBackground ? 'Searching public sources…' : 'Inspecting retrieved support…')
                  : rows.length > 0
                    ? `${rows.length} sources · ${citedCount} cited`
                    : 'Inspect citations, snippets, and supporting pages for the active answer.'}
              </div>
            </div>
            {!loading && rows.length > 0 && (
              <div className="ev-scope-toggle">
                <button className={showCitedOnly ? 'active' : ''} onClick={() => setShowCitedOnly(true)}>Cited</button>
                <button className={!showCitedOnly ? 'active' : ''} onClick={() => setShowCitedOnly(false)}>All</button>
              </div>
            )}
          </div>
        </div>

        <div className="evidence-body">
          {loading ? (
            <>
              <div style={{ padding: '6px 2px 12px', color: 'var(--text-2)', fontSize: 12 }}>
                {allowGeneralBackground ? 'Searching public sources…' : 'Retrieving evidence…'}
              </div>
              <div className="ev-skeleton-list">
                {[0, 1, 2].map((n) => (
                  <div key={n} className="ev-skeleton-card" style={{ animationDelay: `${n * 80}ms` }}>
                    <div className="sk-row sk-full" />
                    <div className="sk-row sk-3q" />
                    <div className="sk-row sk-half" />
                  </div>
                ))}
              </div>
            </>
          ) : visible.length === 0 ? (
            <div className="evidence-empty">
              <div className="evidence-empty-icon">⬡</div>
              <div className="evidence-empty-text">
                Click any assistant reply to inspect its grounding sources here.
              </div>
            </div>
          ) : (
            visible.map((row, idx) => (
              <SourceCard key={`${row.id}-${idx}`} row={row} idx={idx} onClick={() => setModalRow(row)} />
            ))
          )}
        </div>

      </div>
      <SourceModal row={modalRow} onClose={() => setModalRow(null)} />
    </>
  );
}

// ── Follow-up query helpers ───────────────────────────────────────────────────
function isFollowUp(text: string): boolean {
  const q = text.trim().toLowerCase();
  if (!q) return false;
  const cues = [
    'from ieee', 'from springer', 'from elsevier', 'from arxiv',
    'that one', 'this one', 'give me papers', 'relevant papers', 'more info',
  ];
  return q.split(/\s+/).length <= 8 || cues.some((c) => q.includes(c));
}

function enrichQuery(current: string, msgs: UiMessage[]): string {
  const q = current.trim();
  if (!isFollowUp(q)) return q;
  const wantsPapers = /\b(papers?|research|studies|references?|surveys?)\b/i.test(q);
  for (let i = msgs.length - 1; i >= 0; i--) {
    const m = msgs[i];
    if (m.role !== 'you' || !m.text.trim()) continue;
    if (m.text.toLowerCase() === q.toLowerCase()) continue;
    if (!isFollowUp(m.text) && m.text.split(/\s+/).length >= 5) {
      return wantsPapers
        ? `Find relevant research papers about ${m.text}. Include foundational, survey, and highly relevant papers.`
        : `${q} about ${m.text}`;
    }
  }
  return q;
}

function isExplicitDocumentQuery(text: string): boolean {
  const q = text.trim().toLowerCase();
  if (!q) return false;
  return (
    /\b(this|these|that|those|selected|uploaded|attached|active|current)\s+(doc|docs|document|documents|file|files|pdf|pdfs)\b/.test(q) ||
    /\bin\s+(this|these|that|those|the)\s+(doc|docs|document|documents|file|files|pdf|pdfs)\b/.test(q) ||
    /\b(across|from|about)\s+(these|those|selected|uploaded|attached)\s+(doc|docs|documents|files|pdfs)\b/.test(q)
  );
}

function isGreetingQuery(text: string): boolean {
  const q = text.trim().toLowerCase();
  return /^(hi|hello|hey|yo|sup|wassup|what'?s up|howdy|good morning|good afternoon|good evening|hola)[!.?]*$/.test(q);
}

function isAssistantSetupQuery(text: string): boolean {
  const q = text.trim().toLowerCase();
  return (
    /\b(what can you do|how can you help)\b/.test(q) ||
    /\b(help me with research|research based questions|research questions)\b/.test(q) ||
    /\b(answer my research based questions)\b/.test(q)
  );
}

function isLiteratureQuery(text: string): boolean {
  const q = text.trim().toLowerCase();
  return (
    /\b(show me papers|find papers|give me papers|list papers|relevant papers|sources|citations|bibliography)\b/.test(q) ||
    /\b(in the literature|recent papers|recent research|what do papers say)\b/.test(q)
  );
}

function isDocumentGroundedModeQuery(text: string, hasSelectedDocs: boolean): boolean {
  const q = text.trim().toLowerCase();
  if (!q) return false;
  if (!hasSelectedDocs) return false;
  if (isLiteratureQuery(q)) return false;
  if (isExplicitDocumentQuery(q)) return true;
  // When documents are selected, default to using them as the active context
  // unless the user explicitly asks for literature/public-paper discovery.
  return true;
}

function buildAssistantIntroReply(kind: 'greeting' | 'setup', hasUploads: boolean): string {
  const intro = kind === 'greeting'
    ? 'Hi. I can help with concept explanations, paper discovery, research synthesis, and document-grounded analysis.'
    : 'I can explain concepts, find papers, compare findings across the literature, and analyze uploaded documents.';
  const thirdPrompt = hasUploads
    ? 'Select a document and ask for key findings or supporting evidence'
    : 'Upload a paper and ask for a grounded summary';
  return `${intro}

Try one of these:
- Tell me about RNNs
- Show me papers on attention mechanisms
- ${thirdPrompt}`;
}

// ── Eval page ─────────────────────────────────────────────────────────────────
function EvalPage({ onBack }: { onBack: () => void }) {
  const [name, setName] = useState('Local eval run');
  const [k, setK] = useState(10);
  const [rawCases, setRawCases] = useState('[\n  {"query":"DES key size", "expected_doc_id": 48}\n]');
  const [running, setRunning] = useState(false);
  const [presetLoading, setPresetLoading] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState<EvalRunResponse | null>(null);
  const [runs, setRuns] = useState<EvalRunResponse[]>([]);

  const [judgeScope, setJudgeScope] = useState<'uploaded' | 'public'>('uploaded');
  const [judgeK, setJudgeK] = useState(10);
  const [judgeRaw, setJudgeRaw] = useState('[\n  {"query":"What is the main contribution?", "scope":"uploaded"}\n]');
  const [judgeRunning, setJudgeRunning] = useState(false);
  const [judgePresetLoading, setJudgePresetLoading] = useState(false);
  const [judgeError, setJudgeError] = useState('');
  const [judgeResult, setJudgeResult] = useState<JudgeRunResponse | null>(null);
  const [judgeRuns, setJudgeRuns] = useState<JudgeRunSummary[]>([]);

  const [calibRaw, setCalibRaw] = useState('{\n  "records": [\n    {"sentence":"The model answers from evidence","evidence":"chunk contains the claim","M":0.82,"S":0.75,"A":0.70,"label":"strong"}\n  ]\n}');
  const [calibRunning, setCalibRunning] = useState(false);
  const [calibError, setCalibError] = useState('');
  const [calibResult, setCalibResult] = useState<MsaCalibrationResponse | null>(null);
  const [calibLatest, setCalibLatest] = useState<MsaCalibrationLatest | null>(null);

  useEffect(() => {
    api.listEvalRuns(20).then((r) => setRuns(r.runs || [])).catch(() => undefined);
    api.listJudgeRuns(20).then((r) => setJudgeRuns(r.runs || [])).catch(() => undefined);
    api.getLatestCalibration().then((r) => setCalibLatest(r || null)).catch(() => undefined);
  }, []);

  const loadPresetDocs = async (): Promise<DocumentRow[]> => {
    const res = await api.listDocs(100);
    const selected = selectPresetDocuments(res.documents || []);
    if (!selected.length) {
      throw new Error('No ready uploaded documents found for preset generation.');
    }
    return selected;
  };

  const loadEvalPreset = async () => {
    setError(''); setPresetLoading(true);
    try {
      const docs = await loadPresetDocs();
      const cases = buildRetrievalPresetCases(docs, DEFAULT_EVAL_PRESET_COUNT);
      if (!cases.length) throw new Error('Unable to build retrieval cases from the current documents.');
      setName(`Uploaded ${cases.length}-query eval`);
      setRawCases(JSON.stringify(cases, null, 2));
    } catch (e: any) { setError(e?.message || 'Failed to load retrieval preset'); }
    finally { setPresetLoading(false); }
  };

  const loadJudgePreset = async () => {
    setJudgeError(''); setJudgePresetLoading(true);
    try {
      const docs = await loadPresetDocs();
      const cases = buildJudgePresetCases(docs, DEFAULT_EVAL_PRESET_COUNT);
      if (!cases.length) throw new Error('Unable to build judge cases from the current documents.');
      setJudgeScope('uploaded');
      setJudgeRaw(JSON.stringify(cases, null, 2));
    } catch (e: any) { setJudgeError(e?.message || 'Failed to load judge preset'); }
    finally { setJudgePresetLoading(false); }
  };

  const runEval = async () => {
    setError(''); setRunning(true);
    try {
      const cases = JSON.parse(rawCases) as EvalCase[];
      const res = await api.runEval({ name, scope: 'uploaded', k, cases });
      setResult(res);
      const r = await api.listEvalRuns(20);
      setRuns(r.runs || []);
    } catch (e: any) { setError(e?.message || 'Failed'); }
    finally { setRunning(false); }
  };

  const runJudge = async () => {
    setJudgeError(''); setJudgeRunning(true);
    try {
      const cases = JSON.parse(judgeRaw) as JudgeCasePayload[];
      const res = await api.runJudge({ scope: judgeScope, k: judgeK, run_judge_llm: true, cases });
      setJudgeResult(res);
      const r = await api.listJudgeRuns(20);
      setJudgeRuns(r.runs || []);
    } catch (e: any) { setJudgeError(e?.message || 'Failed'); }
    finally { setJudgeRunning(false); }
  };

  const runCalib = async () => {
    setCalibError(''); setCalibRunning(true);
    try {
      const parsed = JSON.parse(calibRaw) as any;
      const records = Array.isArray(parsed) ? parsed : (parsed?.records || null);
      if (!records?.length) throw new Error('records array required');
      const payload: MsaCalibrationPayload = { model_name: 'msa_manual', label: 'manual', ...parsed, records };
      const res = await api.calibrateConfidence(payload);
      setCalibResult(res);
      const latest = await api.getLatestCalibration();
      setCalibLatest(latest);
    } catch (e: any) { setCalibError(e?.message || 'Failed'); }
    finally { setCalibRunning(false); }
  };

  return (
    <div className="eval-page">
      <div className="eval-topbar">
        <h1>Evaluation Studio</h1>
        <button className="btn btn-ghost btn-sm" onClick={onBack}>← Back to Chat</button>
      </div>

      <div className="eval-grid">
        <div className="eval-card">
          <h3>LLM Judge Evaluation</h3>
          <label>Scope</label>
          <select value={judgeScope} onChange={(e) => setJudgeScope(e.target.value as 'uploaded' | 'public')}>
            <option value="uploaded">Uploaded</option>
            <option value="public">Public</option>
          </select>
          <label>Top K</label>
          <input type="number" value={judgeK} onChange={(e) => setJudgeK(Number(e.target.value) || 10)} />
          <label>Test cases JSON</label>
          <div className="eval-actions">
            <button className="btn btn-ghost btn-sm" onClick={loadJudgePreset} disabled={judgeRunning || judgePresetLoading}>
              {judgePresetLoading ? 'Loading preset…' : `Load ${DEFAULT_EVAL_PRESET_COUNT}-query synthesis preset`}
            </button>
            <span className="eval-hint">Builds uploaded-scope single-doc and multi-doc synthesis prompts from ready docs.</span>
          </div>
          <textarea rows={8} value={judgeRaw} onChange={(e) => setJudgeRaw(e.target.value)} />
          <button className="btn btn-primary btn-sm" onClick={runJudge} disabled={judgeRunning || judgePresetLoading}>
            {judgeRunning ? 'Running…' : 'Run judge eval'}
          </button>
          {judgeError && <div className="alert">{judgeError}</div>}
          {judgeResult && (
            <div style={{ fontSize: 12, color: 'var(--text-2)', marginTop: 4, lineHeight: 1.8 }}>
              Mean score: <strong style={{ color: 'var(--text)' }}>{Math.round((judgeResult.metrics.mean_overall_score || 0) * 100)}%</strong>
              {' · '}Unsupported: <strong style={{ color: 'var(--text)' }}>{judgeResult.metrics.unsupported_total || 0}</strong>
            </div>
          )}
        </div>

        <div className="eval-card">
          <h3>M/S/A Calibration</h3>
          <label>Calibration records JSON</label>
          <textarea rows={12} value={calibRaw} onChange={(e) => setCalibRaw(e.target.value)} />
          <button className="btn btn-primary btn-sm" onClick={runCalib} disabled={calibRunning}>
            {calibRunning ? 'Calibrating…' : 'Fit MSA calibration'}
          </button>
          {calibError && <div className="alert">{calibError}</div>}
          {calibResult && (
            <div style={{ fontSize: 12, color: 'var(--text-2)', lineHeight: 1.8 }}>
              <div>{calibResult.model_name} · {calibResult.records_used} records used</div>
              <div>Brier: {calibResult.metrics.brier} · Accuracy: {calibResult.metrics.accuracy}</div>
            </div>
          )}
          {calibLatest && (
            <div style={{ fontSize: 11, color: 'var(--text-3)', borderTop: '1px solid var(--border)', paddingTop: 8 }}>
              Latest: {calibLatest.model_name} ({calibLatest.label}) · {calibLatest.dataset_size} records
            </div>
          )}
        </div>

        <div className="eval-card">
          <h3>Retrieval Evaluation</h3>
          <label>Run name</label>
          <input value={name} onChange={(e) => setName(e.target.value)} />
          <label>Top K</label>
          <input type="number" value={k} onChange={(e) => setK(Number(e.target.value) || 10)} />
          <label>Test cases JSON</label>
          <div className="eval-actions">
            <button className="btn btn-ghost btn-sm" onClick={loadEvalPreset} disabled={running || presetLoading}>
              {presetLoading ? 'Loading preset…' : `Load ${DEFAULT_EVAL_PRESET_COUNT}-query retrieval preset`}
            </button>
            <span className="eval-hint">Anchors each query to the uploaded document title and expected doc id.</span>
          </div>
          <textarea rows={8} value={rawCases} onChange={(e) => setRawCases(e.target.value)} />
          <button className="btn btn-primary btn-sm" onClick={runEval} disabled={running || presetLoading}>
            {running ? 'Running…' : 'Run evaluation'}
          </button>
          {error && <div className="alert">{error}</div>}
        </div>

        <div className="eval-card">
          <h3>Latest result</h3>
          {!result ? (
            <p style={{ color: 'var(--text-3)', fontSize: 13 }}>No run yet.</p>
          ) : (
            <>
              <table className="eval-table">
                <thead>
                  <tr><th>Metric</th><th>Retrieval</th><th>+Rerank</th></tr>
                </thead>
                <tbody>
                  {(['1', '3', '5', '10'] as const).map((n) => (
                    <tr key={n}>
                      <td>Recall@{n}</td>
                      <td>{result.metrics_retrieval_only.recall_at[n]?.toFixed(3) ?? '–'}</td>
                      <td>{result.metrics_retrieval_rerank.recall_at[n]?.toFixed(3) ?? '–'}</td>
                    </tr>
                  ))}
                  <tr>
                    <td>MRR</td>
                    <td>{result.metrics_retrieval_only.mrr.toFixed(3)}</td>
                    <td>{result.metrics_retrieval_rerank.mrr.toFixed(3)}</td>
                  </tr>
                </tbody>
              </table>
              <div className="lat-bars">
                {([
                  ['Retrieve', result.latency_breakdown.retrieve_ms_avg],
                  ['Rerank', result.latency_breakdown.rerank_ms_avg],
                  ['Generate', result.latency_breakdown.generate_ms_avg],
                ] as [string, number][]).map(([label, val]) => (
                  <div key={label} className="lat-row">
                    <span className="lat-label">{label} {Math.round(val)} ms</span>
                    <div className="lat-bar">
                      <div className="lat-fill" style={{ width: `${Math.min(100, val / 3000 * 100)}%` }} />
                    </div>
                  </div>
                ))}
              </div>
            </>
          )}
        </div>
      </div>

      <div className="eval-card" style={{ marginBottom: 16 }}>
        <h3>Stored eval runs</h3>
        <div className="run-list">
          {runs.length === 0
            ? <div style={{ color: 'var(--text-3)', fontSize: 13 }}>No runs yet.</div>
            : runs.map((r) => (
              <div key={`${r.run_id}-${r.created_at}`} className="run-item">
                <strong>{r.name}</strong>
                <span>{r.created_at}</span>
                <span>{r.case_count} cases</span>
                <span>R@5 {r.metrics_retrieval_rerank?.recall_at?.['5'] ?? '–'}</span>
              </div>
            ))}
        </div>
      </div>

      <div className="eval-card">
        <h3>Judge runs</h3>
        <div className="run-list">
          {judgeRuns.length === 0
            ? <div style={{ color: 'var(--text-3)', fontSize: 13 }}>No judge runs yet.</div>
            : judgeRuns.map((r, idx) => (
              <div key={`${r.id}-${idx}`} className="run-item">
                <strong>{r.scope || judgeScope}</strong>
                <span>Run {r.id}</span>
                <span>{r.query_count || 0} queries</span>
                <span>Mean {Math.round(((r.metrics?.mean_overall_score) || 0) * 100)}%</span>
              </div>
            ))}
        </div>
      </div>
    </div>
  );
}

// ── Studio page ───────────────────────────────────────────────────────────────
const STORAGE_KEY = 'scholarrag_studio_v3';
const MEMORY_PREF_KEY = 'scholarrag_memory_pref_v1';
const DEFAULT_SESSION_TITLE = 'New chat';
const EMPTY_EVIDENCE: EvidenceState = { citations: [], trace: [] };

function createSessionId(): string {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID();
  }
  return `chat_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
}

function shortTitle(text: string): string {
  const compact = text.replace(/\s+/g, ' ').trim();
  if (!compact) return DEFAULT_SESSION_TITLE;
  return compact.length > 44 ? `${compact.slice(0, 44).trimEnd()}…` : compact;
}

function deriveSessionTitle(messages: UiMessage[]): string {
  const firstUser = messages.find((m) => m.role === 'you' && m.text.trim());
  return firstUser ? shortTitle(firstUser.text) : DEFAULT_SESSION_TITLE;
}

function createStudioSession(overrides: Partial<StudioSession> = {}): StudioSession {
  const now = Date.now();
  const messages = overrides.messages || [];
  return {
    id: overrides.id || createSessionId(),
    title: overrides.title || deriveSessionTitle(messages),
    messages,
    selectedDocs: overrides.selectedDocs || [],
    activeEvidence: overrides.activeEvidence || EMPTY_EVIDENCE,
    activeEvidenceMsgIdx: overrides.activeEvidenceMsgIdx ?? -1,
    allowGeneralBackground: overrides.allowGeneralBackground ?? false,
    createdAt: overrides.createdAt ?? now,
    updatedAt: overrides.updatedAt ?? now,
  };
}

const INITIAL_SESSION = createStudioSession();

function resolveUpdater<T>(next: T | ((prev: T) => T), prev: T): T {
  return typeof next === 'function' ? (next as (prev: T) => T)(prev) : next;
}

function StudioPage({
  onNavigateEval,
}: {
  onNavigateEval: () => void;
}) {
  const [docs, setDocs] = useState<DocumentRow[]>([]);
  const [sessions, setSessions] = useState<StudioSession[]>([INITIAL_SESSION]);
  const [activeSessionId, setActiveSessionId] = useState<string>(INITIAL_SESSION.id);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [pendingDelete, setPendingDelete] = useState<DocumentRow | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<{ text: string; state: 'idle' | 'uploading' | 'done' | 'err' }>({ text: '', state: 'idle' });
  const [uploadPct, setUploadPct] = useState(0);
  const [showSettings, setShowSettings] = useState(false);
  const [preserveMemory, setPreserveMemory] = useState<boolean>(() => {
    try {
      return localStorage.getItem(MEMORY_PREF_KEY) !== 'off';
    } catch {
      return true;
    }
  });

  const [sidebarTab, setSidebarTab] = useState<'chats' | 'library'>('library');
  const [pendingBatchDelete, setPendingBatchDelete] = useState(false);
  const [diagnosticsMsgIdx, setDiagnosticsMsgIdx] = useState<number | null>(null);
  const [evalSnapshot, setEvalSnapshot] = useState<{
    retrievalR5: number | null;
    judgeMean: number | null;
    updatedAt: string | null;
  }>({
    retrievalR5: null,
    judgeMean: null,
    updatedAt: null,
  });

  const chatEndRef = useRef<HTMLDivElement>(null);
  const chatStreamRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const settingsPopoverRef = useRef<HTMLDivElement>(null);
  const activeSession = useMemo(
    () => sessions.find((session) => session.id === activeSessionId) || sessions[0] || INITIAL_SESSION,
    [sessions, activeSessionId],
  );
  const messages = activeSession.messages;
  const selectedDocs = activeSession.selectedDocs;
  const allowGeneralBackground = activeSession.allowGeneralBackground;
  const activeEvidence = activeSession.activeEvidence;
  const activeEvidenceMsgIdx = activeSession.activeEvidenceMsgIdx;

  const updateActiveSession = (updater: (session: StudioSession) => StudioSession) => {
    setSessions((prev) =>
      prev.map((session) =>
        session.id === activeSessionId
          ? { ...updater(session), updatedAt: Date.now() }
          : session,
      ),
    );
  };

  const setMessages = (next: UiMessage[] | ((prev: UiMessage[]) => UiMessage[])) => {
    updateActiveSession((session) => {
      const messages = resolveUpdater(next, session.messages);
      return {
        ...session,
        messages,
        title: deriveSessionTitle(messages),
      };
    });
  };

  const setSelectedDocs = (next: number[] | ((prev: number[]) => number[])) => {
    updateActiveSession((session) => ({
      ...session,
      selectedDocs: resolveUpdater(next, session.selectedDocs),
    }));
  };

  const setAllowGeneralBackground = (next: boolean | ((prev: boolean) => boolean)) => {
    updateActiveSession((session) => ({
      ...session,
      allowGeneralBackground: resolveUpdater(next, session.allowGeneralBackground),
    }));
  };

  const setActiveEvidence = (next: EvidenceState | ((prev: EvidenceState) => EvidenceState)) => {
    updateActiveSession((session) => ({
      ...session,
      activeEvidence: resolveUpdater(next, session.activeEvidence),
    }));
  };

  const setActiveEvidenceMsgIdx = (next: number | ((prev: number) => number)) => {
    updateActiveSession((session) => ({
      ...session,
      activeEvidenceMsgIdx: resolveUpdater(next, session.activeEvidenceMsgIdx),
    }));
  };

  // Restore session
  useEffect(() => {
    if (!preserveMemory) return;
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) return;
      const s = JSON.parse(raw);
      if (Array.isArray(s.sessions) && s.sessions.length) {
        const restored = s.sessions.map((session: any) => createStudioSession({
          id: session.id,
          title: session.title,
          messages: Array.isArray(session.messages) ? session.messages : [],
          selectedDocs: Array.isArray(session.selectedDocs) ? session.selectedDocs : [],
          activeEvidence: session.activeEvidence || EMPTY_EVIDENCE,
          activeEvidenceMsgIdx: typeof session.activeEvidenceMsgIdx === 'number' ? session.activeEvidenceMsgIdx : -1,
          allowGeneralBackground: Boolean(session.allowGeneralBackground),
          createdAt: session.createdAt,
          updatedAt: session.updatedAt,
        }));
        setSessions(restored);
        setActiveSessionId(
          restored.some((session: StudioSession) => session.id === s.activeSessionId)
            ? s.activeSessionId
            : restored[0].id,
        );
        return;
      }
      if (Array.isArray(s.messages)) {
        const migrated = createStudioSession({
          messages: s.messages,
          selectedDocs: Array.isArray(s.selectedDocs) ? s.selectedDocs : [],
          activeEvidence: s.activeEvidence || EMPTY_EVIDENCE,
          activeEvidenceMsgIdx: typeof s.activeEvidenceMsgIdx === 'number' ? s.activeEvidenceMsgIdx : -1,
        });
        setSessions([migrated]);
        setActiveSessionId(migrated.id);
      }
    } catch {
      // localStorage can be unavailable (privacy mode / quota issues).
    }
  }, [preserveMemory]);

  // Persist session
  useEffect(() => {
    if (!preserveMemory) return;
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify({ sessions, activeSessionId }));
    } catch {
      // localStorage can be unavailable (privacy mode / quota issues).
    }
  }, [sessions, activeSessionId, preserveMemory]);

  useEffect(() => {
    try {
      localStorage.setItem(MEMORY_PREF_KEY, preserveMemory ? 'on' : 'off');
      if (!preserveMemory) {
        localStorage.removeItem(STORAGE_KEY);
      }
    } catch {
      // localStorage can be unavailable (privacy mode / quota issues).
    }
  }, [preserveMemory]);

  const refreshDocs = async () => {
    try {
      const res = await api.listDocs();
      const list = res.documents || [];
      setDocs(list);
      setSessions((prev) =>
        prev.map((session) => ({
          ...session,
          selectedDocs: session.selectedDocs.filter((id) => list.some((d) => d.id === id)),
        })),
      );
      setError('');
    } catch (e: any) {
      setError(e?.message || `Backend unreachable at ${API_BASE}`);
    }
  };

  useEffect(() => { refreshDocs(); }, []);

  useEffect(() => {
    let mounted = true;
    Promise.all([
      api.listEvalRuns(1),
      api.listJudgeRuns(1),
    ])
      .then(([evalRuns, judgeRuns]) => {
        if (!mounted) return;
        const latestEval = evalRuns.runs?.[0];
        const latestJudge = judgeRuns.runs?.[0];
        const retrievalR5Raw = latestEval?.metrics_retrieval_rerank?.recall_at?.['5'];
        const judgeMeanRaw = latestJudge?.metrics?.mean_overall_score;
        setEvalSnapshot({
          retrievalR5: Number.isFinite(retrievalR5Raw) ? Number(retrievalR5Raw) : null,
          judgeMean: Number.isFinite(judgeMeanRaw) ? Number(judgeMeanRaw) : null,
          updatedAt: latestEval?.created_at || latestJudge?.created_at || null,
        });
      })
      .catch(() => undefined);
    return () => {
      mounted = false;
    };
  }, []);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        textareaRef.current?.focus();
        return;
      }
      if ((e.ctrlKey || e.metaKey) && e.key === ',') {
        e.preventDefault();
        setShowSettings((value) => !value);
        return;
      }
      if (e.key === 'Escape') {
        if (showSettings) {
          setShowSettings(false);
          return;
        }
        if (document.activeElement === textareaRef.current) {
          setInput('');
        }
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [showSettings]);

  useEffect(() => {
    if (!showSettings) return;
    const handlePointerDown = (event: MouseEvent) => {
      const target = event.target as Node | null;
      if (!target) return;
      if (settingsPopoverRef.current?.contains(target)) return;
      setShowSettings(false);
    };
    document.addEventListener('mousedown', handlePointerDown);
    return () => document.removeEventListener('mousedown', handlePointerDown);
  }, [showSettings]);

  useEffect(() => {
    if (!docs.some((d) => d.status === 'processing')) return;
    const id = setInterval(refreshDocs, 2500);
    return () => clearInterval(id);
  }, [docs]);

  useEffect(() => {
    setDiagnosticsMsgIdx(null);
    setShowSettings(false);
  }, [activeSessionId]);

  const prevMsgCountRef = useRef<number>(0);
  useEffect(() => {
    const prevCount = prevMsgCountRef.current;
    const currCount = messages.length;
    prevMsgCountRef.current = currCount;

    if (currCount === 0) {
      chatStreamRef.current?.scrollTo({ top: 0 });
    } else if (currCount > prevCount && prevCount > 0) {
      // Only auto-scroll when new messages arrive, not on session restore
      chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    } else if (loading && currCount === prevCount) {
      // Scroll when loading indicator appears for current session
      chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages, loading]);

  const dedupedDocs = useMemo(() => {
    const seen = new Set<string>();
    return docs.filter((d) => {
      const k = d.title.toLowerCase();
      if (seen.has(k)) return false;
      seen.add(k); return true;
    });
  }, [docs]);

  const selectedRows = useMemo(() => dedupedDocs.filter((d) => selectedDocs.includes(d.id)), [dedupedDocs, selectedDocs]);
  const activeDoc = selectedRows[0] || null;
  const hasMulti = selectedRows.length > 1;
  const processedCount = dedupedDocs.filter((d) => d.status === 'ready').length;
  const processingCount = dedupedDocs.filter((d) => d.status === 'processing').length;
  const orderedSessions = useMemo(
    () => [...sessions].sort((a, b) => b.updatedAt - a.updatedAt),
    [sessions],
  );
  const selectedPreview = selectedRows.slice(0, 3);

  const handleFiles = async (files: FileList | null) => {
    if (!files?.length) return;
    setUploadStatus({ text: `Uploading ${files.length} file(s)…`, state: 'uploading' });
    setUploadPct(0);
    try {
      for (let i = 0; i < files.length; i++) {
        setUploadStatus({ text: `Uploading ${i + 1}/${files.length}: ${files[i].name}`, state: 'uploading' });
        setUploadPct(Math.round(((i + 0.5) / files.length) * 100));
        await api.uploadFile(files[i]);
      }
      setUploadPct(100);
      setUploadStatus({
        text: `${files.length === 1 ? files[0].name : `${files.length} files`} uploaded`,
        state: 'done',
      });
      refreshDocs();
      setTimeout(() => setUploadStatus({ text: '', state: 'idle' }), 3000);
    } catch (e: any) {
      setUploadStatus({ text: e?.message || 'Upload failed', state: 'err' });
    }
  };

  const streamMessage = (
    fullText: string,
    meta: Pick<UiMessage, 'citations' | 'confidence' | 'why_answer' | 'latency_breakdown_ms' | 'needs_clarification' | 'clarification' | 'answer_scope' | 'unsupported_claims' | 'query_ref' | 'faithfulness' | 'retrieval_policy'>,
  ) => {
    const text = fullText || 'No response received. Check backend/OpenAI key.';
    setMessages((prev) => [...prev, { role: 'assistant', text, streaming: false, ...meta }]);
    setActiveEvidence({ citations: meta.citations || [], trace: meta.why_answer?.top_chunks || [] });
  };

  const startNewChat = () => {
    const session = createStudioSession({
      selectedDocs,
      allowGeneralBackground,
    });
    setSessions((prev) => [session, ...prev]);
    setActiveSessionId(session.id);
    setInput('');
    setError('');
    setLoading(false);
  };

  const deleteSession = (sessionId: string) => {
    setSessions((prev) => {
      const next = prev.filter((s) => s.id !== sessionId);
      if (next.length === 0) {
        const fresh = createStudioSession();
        setActiveSessionId(fresh.id);
        return [fresh];
      }
      if (sessionId === activeSessionId) {
        setActiveSessionId(next[0].id);
      }
      return next;
    });
  };

  const handleBatchDelete = async () => {
    if (!selectedDocs.length) return;
    const toDelete = [...selectedDocs];
    setPendingBatchDelete(false);
    setSelectedDocs([]);
    setActiveEvidence({ citations: [], trace: [] });
    await Promise.all(toDelete.map((id) => api.deleteDoc(id).catch(() => undefined)));
    refreshDocs();
  };

  const ask = async (text: string, skipEnrichment = false, sense?: string) => {
    const q = text.trim();
    if (!q) return;
    setError('');
    setLoading(true);
    setMessages((prev) => [...prev, { role: 'you', text: q }]);
    setInput('');
    if (textareaRef.current) textareaRef.current.style.height = 'auto';

    try {
      const hasUploads = Boolean(selectedDocs.length || docs.length);
      if (isGreetingQuery(q)) {
        streamMessage(
          buildAssistantIntroReply('greeting', hasUploads),
          { citations: [] },
        );
        setLoading(false);
        return;
      }
      if (selectedDocs.length === 0 && isAssistantSetupQuery(q)) {
        streamMessage(
          buildAssistantIntroReply('setup', hasUploads),
          { citations: [] },
        );
        setLoading(false);
        return;
      }
      const hasSelectedDocs = selectedDocs.length > 0;
      if (!allowGeneralBackground) {
        if (!hasSelectedDocs) {
          streamMessage(
            'Docs only mode is active. Select one or more uploaded documents for grounded answers, or switch to Public research for general research questions.',
            { citations: [] },
          );
          setLoading(false);
          return;
        }
        if (!isDocumentGroundedModeQuery(q, hasSelectedDocs)) {
          streamMessage(
            'Docs only mode is active. Ask about the selected document(s), or switch to Public research for papers, general concepts, or literature search.',
            { citations: [] },
          );
          setLoading(false);
          return;
        }
      }
      const forcePublicWithoutSelection = allowGeneralBackground && hasUploads && selectedDocs.length === 0;
      const scope: 'uploaded' | 'public' =
        forcePublicWithoutSelection ? 'public' : (hasUploads ? 'uploaded' : 'public');
      const query = allowGeneralBackground && !skipEnrichment && scope === 'public' ? enrichQuery(q, messages) : q;
      const res = await api.askAssistant({
        query, scope, sense,
        doc_id: scope === 'uploaded' && selectedDocs.length === 1 ? selectedDocs[0] : undefined,
        doc_ids: scope === 'uploaded' && selectedDocs.length > 1 ? selectedDocs : undefined,
        k: 8,
        allow_general_background: allowGeneralBackground,
      });
      streamMessage((res.answer || res.clarification?.question || '').trim(), {
        citations: res.citations || [],
        confidence: res.confidence,
        why_answer: res.why_answer,
        latency_breakdown_ms: res.latency_breakdown_ms,
        needs_clarification: res.needs_clarification,
        clarification: res.clarification,
        answer_scope: res.answer_scope,
        unsupported_claims: res.unsupported_claims,
        faithfulness: res.faithfulness,
        query_ref: query,
        retrieval_policy: res.retrieval_policy,
      });
    } catch (e: any) {
      setError(e?.message || 'Request failed');
      streamMessage(e?.message || `Backend unreachable at ${API_BASE}`, { citations: [] });
    } finally {
      setLoading(false);
    }
  };

  const toggleDoc = (id: number) =>
    setSelectedDocs((prev) => prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]);

  const forgetMemoryNow = () => {
    updateActiveSession((session) => ({
      ...session,
      title: DEFAULT_SESSION_TITLE,
      messages: [],
      activeEvidence: EMPTY_EVIDENCE,
      activeEvidenceMsgIdx: -1,
    }));
    setInput('');
    setError('');
    setLoading(false);
    try {
      localStorage.removeItem(STORAGE_KEY);
    } catch {
      // localStorage can be unavailable (privacy mode / quota issues).
    }
  };

  const confirmDelete = async () => {
    if (!pendingDelete) return;
    try {
      await api.deleteDoc(pendingDelete.id);
      setSelectedDocs((prev) => prev.filter((id) => id !== pendingDelete.id));
      setActiveEvidence({ citations: [], trace: [] });
    } catch (e: any) {
      setError(e?.message || 'Failed to delete document');
    }
    setPendingDelete(null);
    refreshDocs();
  };

  const quickAsk = (prompt: string) => ask(prompt, true);

  const exportChat = () => {
    if (!messages.length) return;
    const title = activeSession.title !== DEFAULT_SESSION_TITLE ? activeSession.title : 'ScholarRAG Chat';
    const lines: string[] = [`# ${title}\n\n*Exported from ScholarRAG — ${new Date().toLocaleDateString()}*\n\n---\n`];
    messages.forEach((m) => {
      if (m.role === 'you') lines.push(`**You:** ${m.text}\n`);
      else lines.push(`**ScholarRAG:** ${m.text}${uniqueSourceCount(m.citations) ? `\n\n*${uniqueSourceCount(m.citations)} source(s) cited*` : ''}\n`);
      lines.push('');
    });
    const blob = new Blob([lines.join('\n')], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${title.replace(/[^a-z0-9]/gi, '_').slice(0, 60)}.md`;
    a.click();
    URL.revokeObjectURL(url);
    setShowSettings(false);
  };

  const clearAllSessions = () => {
    if (!window.confirm('Delete all sessions and start fresh? This cannot be undone.')) return;
    const fresh = createStudioSession();
    setSessions([fresh]);
    setActiveSessionId(fresh.id);
    setInput('');
    setError('');
    setLoading(false);
    setShowSettings(false);
  };




  const hasEvidence = activeEvidence.citations.length > 0 || activeEvidence.trace.length > 0;
  const showEvidencePanel = !allowGeneralBackground || hasEvidence || loading;

  return (
    <div className={`app-shell ${allowGeneralBackground ? 'mode-research' : 'mode-document'}${showEvidencePanel ? '' : ' no-evidence'}`}>
      {/* ── Sidebar ── */}
      <motion.aside className="sidebar" initial={{ x: -40, opacity: 0 }} animate={{ x: 0, opacity: 1 }} transition={{ duration: 0.4, ease: 'easeOut' }}>
        <div className="sidebar-brand">
          <div className="brand-icon" aria-hidden="true">
            <span className="sigil" />
          </div>
          <div className="brand-text">
            <div className="brand-name">ScholarRAG</div>
            <div className="brand-sub">Research assistant</div>
          </div>
        </div>

        <div className="sidebar-body">
          {/* ── Tab nav ── */}
          <div className="sb-tabs">
            <button
              className={`sb-tab${sidebarTab === 'chats' ? ' active' : ''}`}
              onClick={() => setSidebarTab('chats')}
            >
              Chats
              {orderedSessions.length > 0 && <span className="sb-tab-count">{orderedSessions.length}</span>}
            </button>
            <button
              className={`sb-tab${sidebarTab === 'library' ? ' active' : ''}`}
              onClick={() => setSidebarTab('library')}
            >
              Library
              {dedupedDocs.length > 0 && <span className="sb-tab-count">{dedupedDocs.length}</span>}
            </button>
          </div>

          {sidebarTab === 'chats' ? (
            <div className="sb-panel">
              {/* New session + compact stats */}
              <div className="sb-chats-top">
                <button className="sb-new-chat" onClick={startNewChat}>
                  <span className="sb-new-chat-plus">+</span> New session
                </button>
                <div className="sb-stats">
                  <span>{processedCount}d</span>
                  <span className="sb-stats-dot">·</span>
                  <span>{orderedSessions.length}s</span>
                </div>
              </div>

              {/* Session list */}
              <div className="sb-session-list">
                {orderedSessions.map((session) => {
                  const isActive = session.id === activeSessionId;
                  const turnCount = session.messages.filter((m) => m.role === 'you').length;
                  return (
                    <div key={session.id} className={`sb-session${isActive ? ' active' : ''}`}>
                      <button
                        className="sb-session-btn"
                        onClick={() => { setActiveSessionId(session.id); setInput(''); setError(''); setLoading(false); }}
                      >
                        <span className="sb-session-title">{session.title}</span>
                        <span className="sb-session-meta">
                          {turnCount ? `${turnCount} msg${turnCount > 1 ? 's' : ''}` : 'Empty'}
                          {session.selectedDocs.length ? ` · ${session.selectedDocs.length}d` : ''}
                        </span>
                      </button>
                      <button
                        className="sb-session-del"
                        title="Delete session"
                        onClick={(e) => { e.stopPropagation(); deleteSession(session.id); }}
                      >✕</button>
                    </div>
                  );
                })}
              </div>

            </div>
          ) : (
            <div className="sb-panel">
              {/* Upload dropzone */}
              <div
                className={`sb-upload${dragActive ? ' over' : ''}`}
                onDragOver={(e) => { e.preventDefault(); setDragActive(true); }}
                onDragLeave={() => setDragActive(false)}
                onDrop={(e) => { e.preventDefault(); setDragActive(false); handleFiles(e.dataTransfer.files); }}
                onClick={() => document.getElementById('sr-upload')?.click()}
              >
                <input
                  id="sr-upload"
                  type="file"
                  accept=".pdf,.txt,.md"
                  multiple
                  style={{ display: 'none' }}
                  onChange={(e) => handleFiles(e.target.files)}
                />
                {uploadStatus.state === 'uploading' ? (
                  <div className="sb-upload-progress">
                    <div className="sb-upload-bar">
                      <div className="sb-upload-fill" style={{ width: `${uploadPct}%` }} />
                    </div>
                    <span className="sb-upload-label">{uploadStatus.text}</span>
                  </div>
                ) : (
                  <div className="sb-upload-idle">
                    <div className="sb-upload-icon">
                      {uploadStatus.state === 'done' ? '✓' : uploadStatus.state === 'err' ? '!' : '↑'}
                    </div>
                    <span className="sb-upload-label">
                      {uploadStatus.state === 'done' ? uploadStatus.text
                        : uploadStatus.state === 'err' ? uploadStatus.text
                        : 'Drop files or browse'}
                    </span>
                    <span className="sb-upload-hint">PDF · TXT · MD · multiple files</span>
                  </div>
                )}
              </div>

              {/* Library header */}
              {dedupedDocs.length > 0 && (
                <div className="sb-lib-bar">
                  <label className="sb-lib-check-all">
                    <input
                      type="checkbox"
                      checked={dedupedDocs.length > 0 && dedupedDocs.every((d) => selectedDocs.includes(d.id))}
                      onChange={(e) => setSelectedDocs(
                        e.target.checked
                          ? dedupedDocs.filter((d) => d.status === 'ready').map((d) => d.id)
                          : []
                      )}
                    />
                    <span>{dedupedDocs.length} doc{dedupedDocs.length !== 1 ? 's' : ''}</span>
                  </label>
                  {selectedDocs.length > 0 && (
                    <button className="sb-lib-del" onClick={() => setPendingBatchDelete(true)}>
                      Delete {selectedDocs.length}
                    </button>
                  )}
                </div>
              )}

              {/* Doc list */}
              {dedupedDocs.length === 0 ? (
                <div className="sb-empty">
                  <div className="sb-empty-icon">📂</div>
                  <span>No documents yet</span>
                  <span className="sb-empty-hint">Upload a PDF to get started</span>
                </div>
              ) : (
                <div className="sb-doc-list">
                  {dedupedDocs.map((d) => {
                    const sel = selectedDocs.includes(d.id);
                    const isReady = d.status === 'ready';
                    const isError = d.status === 'error';
                    return (
                      <div key={d.id} className={`sb-doc${sel ? ' sel' : ''}`} onClick={() => toggleDoc(d.id)}>
                        <div className={`sb-doc-check${sel ? ' on' : ''}`} />
                        <div className="sb-doc-body">
                          <span className="sb-doc-name" title={d.title}>{d.title}</span>
                          <span className={`sb-doc-status${isReady ? ' ready' : isError ? ' err' : ' proc'}`}>
                            {isReady ? 'Ready' : isError ? 'Error' : 'Processing…'}
                          </span>
                        </div>
                        <button
                          className="sb-doc-del"
                          title="Delete"
                          onClick={(e) => { e.stopPropagation(); setPendingDelete(d); }}
                        >✕</button>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          )}
        </div>

        <div className="sidebar-footer">
          <button className="sidebar-footer-btn" onClick={onNavigateEval}>
            <span className="sidebar-footer-btn-icon">⚗</span>
            Evaluation Studio
          </button>
          <div className="sidebar-footer-metrics" title="Latest retrieval and judge metrics">
            <div className="sidebar-footer-metrics-kicker">Latest Eval Metrics</div>
            <div className="sidebar-footer-metrics-row">
              <span>R@5 {evalSnapshot.retrievalR5 != null ? evalSnapshot.retrievalR5.toFixed(3) : '—'}</span>
              <span>Judge {evalSnapshot.judgeMean != null ? `${Math.round(evalSnapshot.judgeMean * 100)}%` : '—'}</span>
            </div>
            {evalSnapshot.updatedAt ? <div className="sidebar-footer-metrics-time">{evalSnapshot.updatedAt}</div> : null}
          </div>
        </div>
      </motion.aside>

      {/* ── Main content ── */}
      <motion.div className="main-content" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.5, delay: 0.1 }}>
        <div className="slim-bar">
          <div className="slim-bar-left">
            <div className="seg-toggle slim-seg-toggle">
              <button
                className={!allowGeneralBackground ? 'active' : ''}
                onClick={() => setAllowGeneralBackground(false)}
                type="button"
                aria-pressed={!allowGeneralBackground}
              >Docs only</button>
              <button
                className={allowGeneralBackground ? 'active' : ''}
                onClick={() => setAllowGeneralBackground(true)}
                type="button"
                aria-pressed={allowGeneralBackground}
              >Public research</button>
            </div>
            <div className="slim-context-chips">
              {selectedPreview.map((doc) => (
                <button key={doc.id} className="context-chip" onClick={() => toggleDoc(doc.id)}>
                  {doc.title}
                  <span className="context-chip-close">×</span>
                </button>
              ))}
              {selectedRows.length > selectedPreview.length && (
                <span className="context-chip muted">+{selectedRows.length - selectedPreview.length} more</span>
              )}
            </div>
          </div>
          <div className="slim-bar-right">
            <motion.button
              className="slim-btn slim-btn-eval"
              onClick={onNavigateEval}
              type="button"
              whileHover={{ scale: 1.04 }}
              whileTap={{ scale: 0.96 }}
            >
              <svg viewBox="0 0 24 24" width="14" height="14" aria-hidden="true" fill="none"><path d="M9 3H5a2 2 0 00-2 2v4m6-6h10a2 2 0 012 2v4M9 3v18m0 0h10a2 2 0 002-2V9M9 21H5a2 2 0 01-2-2V9m0 0h18" stroke="currentColor" strokeWidth="1.5"/></svg>
              Eval Studio
            </motion.button>
            <div className="workspace-settings" ref={settingsPopoverRef}>
              <button
                className={`slim-btn slim-btn-settings${showSettings ? ' active' : ''}`}
                onClick={() => setShowSettings((value) => !value)}
                type="button"
                aria-expanded={showSettings}
                aria-haspopup="dialog"
                aria-label="Open workspace settings"
              >
                <SettingsIcon />
                Settings
              </button>
              {showSettings && (
                <div className="settings-panel" role="dialog" aria-label="Workspace settings">
                  <div className="settings-header">
                    <span className="settings-header-title">Settings</span>
                    <button className="settings-close" onClick={() => setShowSettings(false)} type="button" aria-label="Close">&times;</button>
                  </div>
                  <div className="settings-body">
                    <div className="settings-group">
                      <div className="settings-row">
                        <div className="settings-row-info">
                          <span className="settings-row-label">Session memory</span>
                          <span className="settings-row-desc">{preserveMemory ? 'Persists across refreshes' : 'Clears when tab closes'}</span>
                        </div>
                        <button
                          className={`settings-toggle${preserveMemory ? ' on' : ''}`}
                          onClick={() => setPreserveMemory((v) => !v)}
                          aria-pressed={preserveMemory}
                          type="button"
                        >
                          <span className="settings-toggle-track"><span className="settings-toggle-thumb" /></span>
                        </button>
                      </div>
                    </div>
                    <div className="settings-group">
                      <span className="settings-group-label">Actions</span>
                      <button className="settings-action" onClick={forgetMemoryNow} type="button">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M3 6h18M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2"/></svg>
                        Clear current chat
                      </button>
                      <button className="settings-action" onClick={exportChat} disabled={messages.length === 0} type="button">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M7 10l5 5 5-5M12 15V3"/></svg>
                        Export as Markdown
                      </button>
                    </div>
                    <div className="settings-group settings-group-danger">
                      <button className="settings-action danger" onClick={clearAllSessions} type="button">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>
                        Clear all sessions
                      </button>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Chat stream */}
        <div className="chat-stream" ref={chatStreamRef}>
          <div className="chat-inner">
            {messages.length === 0 && !loading && (
              <AnimatePresence mode="wait">
                {allowGeneralBackground ? (
                  <motion.div key="public" className="chat-empty mode-public" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.3 }}>
                    <div className="hero-gradient-bg">
                      <MeshGradient className="hero-shader" colors={['#fffbeb', '#f97316', '#fb923c', '#fbbf24', '#fef3c7']} speed={0.12} />
                    </div>
                    <motion.div className="hero-content" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5, delay: 0.1 }}>
                      <motion.span className="hero-badge" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.2 }}>Public Research</motion.span>
                      <motion.h1 className="hero-title" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.25 }}>Explore the open literature</motion.h1>
                      <motion.p className="hero-desc" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.35 }}>Search across Semantic Scholar, OpenAlex, arXiv, and Crossref. Get grounded, cited answers from millions of papers.</motion.p>
                    </motion.div>
                    <div className="empty-grid">
                      {[
                        { eyebrow: 'Explain', title: 'Understand a concept', text: 'Get a clear, cited explanation of any research topic.', prompt: 'Explain how attention mechanisms work in transformers.' },
                        { eyebrow: 'Discover', title: 'Find relevant papers', text: 'Surface the strongest papers, surveys, and references.', prompt: 'Find recent papers on retrieval-augmented generation.' },
                        { eyebrow: 'Compare', title: 'Compare approaches', text: 'Synthesize tradeoffs across methods with source evidence.', prompt: 'Compare BERT, GPT, and T5 architectures.' },
                        { eyebrow: 'Survey', title: 'Literature overview', text: 'Get a structured overview of a research area with key works.', prompt: 'Give me a literature overview of graph neural networks.' },
                      ].map((card, ci) => (
                        <motion.button key={card.eyebrow} className="empty-card" onClick={() => quickAsk(card.prompt)} initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.4 + ci * 0.07 }} whileHover={{ y: -3 }} whileTap={{ scale: 0.98 }}>
                          <div className="empty-card-eyebrow">{card.eyebrow}</div>
                          <div className="empty-card-title">{card.title}</div>
                          <div className="empty-card-text">{card.text}</div>
                        </motion.button>
                      ))}
                    </div>
                  </motion.div>
                ) : (
                  <motion.div key="docs" className="chat-empty mode-docs" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.3 }}>
                    <motion.div className="docs-hero" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5, delay: 0.1 }}>
                      <div className="docs-hero-icon">
                        <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>
                      </div>
                      <motion.span className="hero-badge docs" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.2 }}>Document Analysis</motion.span>
                      <motion.h1 className="hero-title" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.25 }}>
                        {activeDoc ? activeDoc.title : 'Analyze your documents'}
                      </motion.h1>
                      <motion.p className="hero-desc" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.35 }}>
                        {activeDoc
                          ? 'Ask questions grounded entirely in this document\u2019s content. Every answer is backed by specific passages.'
                          : 'Upload a PDF from the sidebar, select it, and ask questions. Answers are grounded only in your document.'}
                      </motion.p>
                    </motion.div>
                    <div className="empty-grid">
                      {(activeDoc || hasMulti ? [
                        { eyebrow: 'Summary', title: hasMulti ? 'Summarize all documents' : 'Summarize this document', text: 'Get the main findings, structure, and key takeaways.', prompt: hasMulti ? 'Summarize the selected uploaded documents.' : 'Summarize the selected uploaded document.' },
                        { eyebrow: 'Key points', title: 'Extract key concepts', text: 'Pull out the strongest claims, topics, and themes.', prompt: 'Extract the key skills, technical topics, and standout claims from the selected uploaded document.' },
                        { eyebrow: 'Evidence', title: 'Inspect evidence', text: 'Surface the passages that best support the main claims.', prompt: 'What evidence best supports the main claims in this document?' },
                        { eyebrow: 'Gaps', title: 'Find gaps and risks', text: 'Identify weak support, ambiguities, or missing details.', prompt: 'Identify weakly supported claims or missing details in this document.' },
                      ] : [
                        { eyebrow: 'Upload', title: 'Upload a document', text: 'Drop a PDF into the sidebar to get started with grounded analysis.', prompt: '' },
                        { eyebrow: 'Summary', title: 'Get a summary', text: 'Once a document is selected, ask for a structured summary.', prompt: '' },
                        { eyebrow: 'Evidence', title: 'Check evidence', text: 'Verify which passages support the document\u2019s claims.', prompt: '' },
                        { eyebrow: 'Gaps', title: 'Spot weaknesses', text: 'Find unsupported assertions and missing methodology details.', prompt: '' },
                      ]).map((card, ci) => (
                        <motion.button key={card.eyebrow} className="empty-card" onClick={() => card.prompt && quickAsk(card.prompt)} initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.4 + ci * 0.07 }} whileHover={{ y: -3 }} whileTap={{ scale: 0.98 }} disabled={!card.prompt}>
                          <div className="empty-card-eyebrow">{card.eyebrow}</div>
                          <div className="empty-card-title">{card.title}</div>
                          <div className="empty-card-text">{card.text}</div>
                        </motion.button>
                      ))}
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            )}

            {error && <div className="alert" style={{ marginBottom: 12 }}>{error}</div>}

            {messages.map((m, i) => {
              const activateEvidence = () => {
                if (m.role !== 'assistant') return;
                setActiveEvidence({ citations: m.citations || [], trace: m.why_answer?.top_chunks || [] });
                setActiveEvidenceMsgIdx(i);
              };
              return (
                <motion.div key={i} className="msg-group" initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.35, delay: 0.05 }}>
                  <div className={`bubble-row ${m.role}`}>
                  {m.role === 'assistant' && (
                    <motion.div className="msg-avatar assistant" aria-hidden="true" initial={{ scale: 0.6, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} transition={{ type: 'spring', stiffness: 300, damping: 22 }}>
                      <span className="sigil" />
                    </motion.div>
                  )}
                  <motion.div
                    className={`msg-bubble${m.role === 'assistant' && activeEvidenceMsgIdx === i ? ' active-evidence' : ''}`}
                    onClick={activateEvidence}
                    initial={m.role === 'you' ? { opacity: 0, x: 20 } : { opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.3 }}
                    whileHover={m.role === 'assistant' ? { y: -1 } : undefined}
                  >
                    {m.role === 'assistant' ? renderMarkdown(m.text) : <div className="md"><p>{m.text}</p></div>}
                    {m.streaming && <span className="stream-cursor" />}
                  </motion.div>
                </div>

                {m.role === 'assistant' && !m.streaming && (
                  <>
                    <InlineCitations citations={m.citations} onActivateEvidence={activateEvidence} />
                    <div className="msg-meta">
                      <div className="assistant-card-head">
                        <span className="assistant-card-label">{formatAnswerScope(m.answer_scope) || 'Answer'}</span>
                        {uniqueSourceCount(m.citations) ? (
                          <span className="assistant-card-count">
                            {uniqueSourceCount(m.citations)} source{uniqueSourceCount(m.citations) > 1 ? 's' : ''}
                          </span>
                        ) : null}
                      </div>
                      <ConfBadge confidence={m.confidence} showWhenMissing={Boolean(m.citations?.length)} />
                      {m.faithfulness && (
                        <span className="meta-chip">
                          Faithfulness: {Math.round((m.faithfulness.overall_score || 0) * 100)}%
                        </span>
                      )}
                      {formatProviderSummary(m) && (
                        <span className="meta-chip scope-chip" title="selected/fetched per provider">
                          {formatProviderSummary(m)}
                        </span>
                      )}
                      <button
                        type="button"
                        className={`meta-chip diagnostics-toggle${diagnosticsMsgIdx === i ? ' active' : ''}`}
                        onClick={() => {
                          activateEvidence();
                          setDiagnosticsMsgIdx((prev) => (prev === i ? null : i));
                        }}
                        title="Show confidence diagnostics"
                      >
                        {diagnosticsMsgIdx === i ? 'Hide diagnostics' : 'Diagnostics'}
                      </button>
                    </div>
                    {diagnosticsMsgIdx === i ? <ConfidenceDiagnostics message={m} /> : null}
                    {m.needs_clarification && m.clarification?.options?.length ? (
                      <div className="clarify-box">
                        <div className="clarify-q">{m.clarification.question}</div>
                        <div className="clarify-opts">
                          {m.clarification.options.map((opt) => (
                            <button
                              key={opt}
                              className="clarify-opt"
                              onClick={() => ask(m.query_ref || '', true, opt)}
                            >
                              {opt}
                            </button>
                          ))}
                        </div>
                      </div>
                    ) : null}
                  </>
                )}
              </motion.div>
              );
            })}

            <AnimatePresence>{loading && <TypingIndicator />}</AnimatePresence>
            <div ref={chatEndRef} />
          </div>

          {/* Quick chips */}
          {messages.length > 0 && !loading && activeDoc && (
            <div className="quick-actions">
              {[
                {
                  label: '📋 Summarize',
                  prompt: hasMulti
                    ? 'Summarize the selected uploaded documents. Give a short section for each document and then a combined takeaway.'
                    : 'Summarize the selected uploaded document.',
                },
                {
                  label: '🔑 Key points',
                  prompt: hasMulti
                    ? 'For each selected uploaded document, extract the key skills, technical topics, standout projects or claims, and one short cross-document comparison at the end.'
                    : 'Extract the key skills, technical topics, standout projects or claims from the selected uploaded document.',
                },
                {
                  label: '🔍 Evidence',
                  prompt: hasMulti
                    ? 'What evidence best supports the main claims in each selected uploaded document? Organize the answer by document.'
                    : 'What evidence best supports the main claims in the selected uploaded document?',
                },
              ].map((c, qi) => (
                <motion.button key={c.label} className="quick-chip" onClick={() => quickAsk(c.prompt)} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: qi * 0.06 }} whileHover={{ y: -2, scale: 1.04 }} whileTap={{ scale: 0.96 }}>
                  {c.label}
                </motion.button>
              ))}
            </div>
          )}
        </div>

        <div className="composer-wrap">
          <PromptInputBox
            value={input}
            onValueChange={setInput}
            onSend={(message) => ask(message)}
            isLoading={loading}
            inputRef={textareaRef}
            contextHint={
              allowGeneralBackground
                ? undefined
                : selectedDocs.length > 0
                  ? `${selectedDocs.length} doc${selectedDocs.length > 1 ? 's' : ''} selected`
                  : 'Docs-only mode: select a document to ground answers'
            }
            placeholder={
              allowGeneralBackground
                ? 'Ask a research question…'
                : selectedDocs.length > 0
                  ? 'Ask about your selected documents…'
                  : 'Select a document to begin, or switch to Public research…'
            }
          />
        </div>
      </motion.div>

      {/* ── Evidence panel ── */}
      <EvidencePanel
        citations={activeEvidence.citations}
        traceChunks={activeEvidence.trace}
        loading={loading}
        allowGeneralBackground={allowGeneralBackground}
      />

      <DeleteModal
        doc={pendingDelete}
        onCancel={() => setPendingDelete(null)}
        onConfirm={confirmDelete}
      />

      {pendingBatchDelete && (
        <div className="modal-backdrop" onClick={() => setPendingBatchDelete(false)}>
          <div className="modal-card" onClick={(e) => e.stopPropagation()}>
            <div className="modal-head">
              <div className="modal-head-title">Delete {selectedDocs.length} document{selectedDocs.length > 1 ? 's' : ''}?</div>
              <button className="modal-close" onClick={() => setPendingBatchDelete(false)}>✕</button>
            </div>
            <div className="modal-body">
              <p>Remove <strong>{selectedDocs.length} selected document{selectedDocs.length > 1 ? 's' : ''}</strong> from your workspace?</p>
              <p style={{ marginTop: 10 }}>All chunks and embeddings will be permanently deleted.</p>
            </div>
            <div className="modal-actions">
              <button className="btn btn-ghost btn-sm" onClick={() => setPendingBatchDelete(false)}>Cancel</button>
              <button className="btn btn-danger btn-sm" onClick={handleBatchDelete}>Delete all</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ── Root with routing ─────────────────────────────────────────────────────────
export default function App() {
  const [page, setPage] = useState<Page>(() =>
    window.location.pathname.startsWith('/eval') ? 'eval' : 'studio',
  );

  useEffect(() => {
    const handler = () =>
      setPage(window.location.pathname.startsWith('/eval') ? 'eval' : 'studio');
    window.addEventListener('popstate', handler);
    return () => window.removeEventListener('popstate', handler);
  }, []);

  useEffect(() => {
    document.documentElement.classList.remove('theme-light');
  }, []);

  useEffect(() => {
    document.body.dataset.page = page;
    return () => {
      delete document.body.dataset.page;
    };
  }, [page]);

  const goEval = () => { window.history.pushState({}, '', '/eval'); setPage('eval'); };
  const goStudio = () => { window.history.pushState({}, '', '/'); setPage('studio'); };

  if (page === 'eval') return <EvalPage onBack={goStudio} />;
  return (
    <StudioPage
      onNavigateEval={goEval}
    />
  );
}
