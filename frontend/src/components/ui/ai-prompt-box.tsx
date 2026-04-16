import {
  useEffect,
  useMemo,
  useRef,
  type KeyboardEvent,
  type MutableRefObject,
} from 'react';
import { ArrowUp, Square } from 'lucide-react';

interface PromptInputBoxProps {
  value: string;
  onValueChange: (value: string) => void;
  onSend: (message: string) => void;
  isLoading?: boolean;
  placeholder?: string;
  disabled?: boolean;
  inputRef?: MutableRefObject<HTMLTextAreaElement | null>;
  contextHint?: string;
}

export function PromptInputBox({
  value,
  onValueChange,
  onSend,
  isLoading = false,
  placeholder = 'Ask a research question...',
  disabled = false,
  inputRef,
  contextHint,
}: PromptInputBoxProps) {
  const localRef = useRef<HTMLTextAreaElement | null>(null);

  useEffect(() => {
    if (!localRef.current) return;
    localRef.current.style.height = 'auto';
    localRef.current.style.height = `${Math.min(localRef.current.scrollHeight, 170)}px`;
  }, [value]);

  const assignRef = (node: HTMLTextAreaElement | null) => {
    localRef.current = node;
    if (inputRef) inputRef.current = node;
  };

  const canSend = useMemo(() => value.trim().length > 0 && !disabled && !isLoading, [value, disabled, isLoading]);

  const handleSubmit = () => {
    const clean = value.trim();
    if (!clean || disabled || isLoading) return;
    onSend(clean);
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="prompt-box">
      {contextHint && (
        <div className="prompt-context">{contextHint}</div>
      )}
      <textarea
        ref={assignRef}
        value={value}
        onChange={(e) => onValueChange(e.target.value)}
        onKeyDown={handleKeyDown}
        className="prompt-textarea"
        placeholder={placeholder}
        disabled={disabled || isLoading}
        rows={1}
      />
      <div className="prompt-footer">
        <button
          type="button"
          onClick={handleSubmit}
          disabled={!canSend}
          className={`prompt-send${canSend ? ' ready' : ''}`}
          title="Send (Enter)"
        >
          {isLoading ? <Square size={16} /> : <ArrowUp size={16} />}
        </button>
      </div>
    </div>
  );
}
