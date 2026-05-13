import { setPipelineMode, type PipelineMode } from '$lib/stores/pipelineMode.svelte';

export type RagMode = PipelineMode | 'chat';
export type RagLogLevel = 'off' | 'summary' | 'full';
export type RagTone = 'neutral' | 'friendly' | 'socratic';
export type RagFormat = 'prose' | 'structured' | 'direct';

export const DEFAULT_TONE: RagTone = 'neutral';
export const DEFAULT_FORMAT: RagFormat = 'prose';

export const TONE_LABELS: Record<RagTone, string> = {
	neutral: 'Neutral',
	friendly: 'Friendly',
	socratic: 'Socratic',
};

export const FORMAT_LABELS: Record<RagFormat, string> = {
	prose: 'Prose',
	structured: 'Structured',
	direct: 'Direct',
};

export const TONE_DESCRIPTIONS: Record<RagTone, string> = {
	neutral: 'Factual, no filler — reads like a reference',
	friendly: 'Plain language, no jargon — like a curious friend',
	socratic: 'Answers then asks a follow-up to deepen understanding',
};

export const FORMAT_DESCRIPTIONS: Record<RagFormat, string> = {
	prose: 'Flowing paragraphs, no bullet points',
	structured: 'Headers and bullets where it helps',
	direct: 'One-line answer first, explanation below',
};

export const RAG_COMMANDS = [
	'/balanced',
	'/chat',
	'/new',
	'/cache',
	'/log off',
	'/log summary',
	'/log full',
	'/think',
	'/no_think',
	'/reset'
] as const;

/** Model families that support a thinking/reasoning mode */
const THINKING_MODEL_PATTERNS = ['qwen3', 'qwq', 'deepseek-r1'];

export function modelSupportsThinking(modelName: string | null | undefined): boolean {
	if (!modelName) return false;
	const lower = modelName.toLowerCase();
	return THINKING_MODEL_PATTERNS.some((p) => lower.includes(p));
}

const ACTIVE_ZIMS_KEY = 'rag_active_zims';
const TONE_KEY = 'rag_tone';
const FORMAT_KEY = 'rag_format';

function loadActiveZims(): string[] {
	try {
		const raw = localStorage.getItem(ACTIVE_ZIMS_KEY);
		return raw ? JSON.parse(raw) : [];
	} catch {
		return [];
	}
}

function saveActiveZims(zims: string[]): void {
	try {
		localStorage.setItem(ACTIVE_ZIMS_KEY, JSON.stringify(zims));
	} catch { /* ignore */ }
}

function loadTone(): RagTone {
	try {
		const raw = localStorage.getItem(TONE_KEY);
		if (raw === 'neutral' || raw === 'friendly' || raw === 'socratic') return raw;
	} catch { /* ignore */ }
	return DEFAULT_TONE;
}

function loadFormat(): RagFormat {
	try {
		const raw = localStorage.getItem(FORMAT_KEY);
		if (raw === 'prose' || raw === 'structured' || raw === 'direct') return raw;
	} catch { /* ignore */ }
	return DEFAULT_FORMAT;
}

function saveTone(tone: RagTone): void {
	try { localStorage.setItem(TONE_KEY, tone); } catch { /* ignore */ }
}

function saveFormat(format: RagFormat): void {
	try { localStorage.setItem(FORMAT_KEY, format); } catch { /* ignore */ }
}

export interface RagControlsState {
	mode: RagMode;
	activeZims: string[];   // empty = all ZIMs active
	bypassCache: boolean;
	logLevel: RagLogLevel;
	thinking: boolean;
	forkActive: boolean;
	tone: RagTone;
	format: RagFormat;
}

const DEFAULT_STATE: RagControlsState = {
	mode: 'chat',
	activeZims: [],
	bypassCache: false,
	logLevel: 'full',
	thinking: false,
	forkActive: false,
	tone: DEFAULT_TONE,
	format: DEFAULT_FORMAT,
};

let _state = $state<RagControlsState>({
	...DEFAULT_STATE,
	activeZims: loadActiveZims(),
	tone: loadTone(),
	format: loadFormat(),
});

function syncPipelineMode(mode: RagMode): void {
	if (mode !== 'chat') setPipelineMode(mode);
}

export function ragControls(): RagControlsState {
	return _state;
}

export function setRagMode(mode: RagMode): void {
	_state = { ..._state, mode };
	syncPipelineMode(mode);
}

export function setActiveZims(zims: string[]): void {
	_state = { ..._state, activeZims: zims };
	saveActiveZims(zims);
}

export function toggleZim(name: string, allZimNames: string[]): void {
	const current = _state.activeZims;
	// empty means all active — expand to full list first
	const expanded = current.length === 0 ? allZimNames : [...current];
	const next = expanded.includes(name)
		? expanded.filter((z) => z !== name)
		: [...expanded, name];
	// if all are selected, collapse back to empty (= all)
	const collapsed = next.length === allZimNames.length ? [] : next;
	setActiveZims(collapsed);
}

export function setBypassCache(bypassCache: boolean): void {
	_state = { ..._state, bypassCache };
}

export function setLogLevel(logLevel: RagLogLevel): void {
	_state = { ..._state, logLevel };
}

export function setTone(tone: RagTone): void {
	_state = { ..._state, tone };
	saveTone(tone);
}

export function setFormat(format: RagFormat): void {
	_state = { ..._state, format };
	saveFormat(format);
}

export function resetStyle(): void {
	setTone(DEFAULT_TONE);
	setFormat(DEFAULT_FORMAT);
}

export function resetRagControls(): void {
	_state = { ...DEFAULT_STATE };
	syncPipelineMode(DEFAULT_STATE.mode);
	saveTone(DEFAULT_TONE);
	saveFormat(DEFAULT_FORMAT);
}



export function resetRagMode(): void {
	setRagMode(DEFAULT_STATE.mode);
}

export function resetActiveZims(): void {
	setActiveZims(DEFAULT_STATE.activeZims);
}

export function resetBypassCache(): void {
	setBypassCache(DEFAULT_STATE.bypassCache);
}

export function resetLogLevel(): void {
	setLogLevel('off');
}

export function setThinking(thinking: boolean): void {
	_state = { ..._state, thinking };
}

export function toggleThinking(): void {
	_state = { ..._state, thinking: !_state.thinking };
}

export function resetThinking(): void {
	setThinking(false);
}

export function setForkActive(active: boolean): void {
	_state = { ..._state, forkActive: active };
}

export function resetForkActive(): void {
	setForkActive(false);
}

function parseLeadingCommand(input: string): {
	handled: boolean;
	message?: string;
	remaining: string;
} {
	const text = input.trimStart();
	const commandMatch = text.match(/^\/([a-zA-Z-]+)(?:\(([^)]*)\))?/);
	if (!commandMatch) return { handled: false, remaining: input };

	const command = commandMatch[1].toLowerCase();
	let arg = (commandMatch[2] ?? '').trim();
	let consumed = commandMatch[0].length;

	function consumeNextToken(): string {
		const rest = text.slice(consumed).trimStart();
		const match = rest.match(/^(\S+)/);
		if (!match) return '';
		consumed = text.length - rest.length + match[0].length;
		return match[0].trim();
	}

	if (command === 'balanced' || command === 'balance') {
		setRagMode('balanced');
		return { handled: true, remaining: text.slice(consumed).trimStart() };
	}
	if (command === 'chat') {
		setRagMode('chat');
		return { handled: true, remaining: text.slice(consumed).trimStart() };
	}
	if (command === 'new' || command === 'cache-new') {
		setBypassCache(true);
		return { handled: true, remaining: text.slice(consumed).trimStart() };
	}
	if (command === 'cache') {
		setBypassCache(false);
		return { handled: true, remaining: text.slice(consumed).trimStart() };
	}
	if (command === 'log') {
		if (!arg) arg = consumeNextToken();
		const level = arg.toLowerCase();
		if (level === 'summary' || level === 'full' || level === 'off') {
			setLogLevel(level);
			return { handled: true, remaining: text.slice(consumed).trimStart() };
		}
		return {
			handled: true,
			message: 'Usage: /log off, /log summary, or /log full',
			remaining: text.slice(consumed).trimStart()
		};
	}
	if (command === 'think') {
		setThinking(true);
		return { handled: true, remaining: text.slice(consumed).trimStart() };
	}
	if (command === 'no_think' || command === 'nothink' || command === 'no-think') {
		setThinking(false);
		return { handled: true, remaining: text.slice(consumed).trimStart() };
	}
	if (command === 'metric' || command === 'metrics') {
		return {
			handled: true,
			message: 'Metrics are shown by default in the answer dropdown.',
			remaining: text.slice(consumed).trimStart()
		};
	}
	if (command === 'reset') {
		resetRagControls();
		return { handled: true, remaining: text.slice(consumed).trimStart() };
	}
	if (command === 'help') {
		return {
			handled: true,
			message:
				'Commands: /balanced, /chat, /new, /cache, /log off|summary|full, /think, /no_think, /reset',
			remaining: text.slice(consumed).trimStart()
		};
	}

	return { handled: false, remaining: input };
}

export function parseRagCommand(input: string): {
	handled: boolean;
	message?: string;
	remaining: string;
} {
	let remaining = input.trim();
	let handled = false;
	let message: string | undefined;

	while (remaining.startsWith('/')) {
		const result = parseLeadingCommand(remaining);
		if (!result.handled) break;
		handled = true;
		message = result.message ?? message;
		remaining = result.remaining.trimStart();
		if (message) break;
	}

	return { handled, message, remaining };
}

export function completeRagCommand(input: string, cursorPosition = input.length): string | null {
	const beforeCursor = input.slice(0, cursorPosition);
	const afterCursor = input.slice(cursorPosition);
	const match = beforeCursor.match(/(^|\s)(\/[a-zA-Z-]*(?:\s+[a-zA-Z-]*)?)$/);
	if (!match) return null;

	const prefix = match[2].toLowerCase();
	if (!prefix.startsWith('/')) return null;

	const matches = RAG_COMMANDS.filter((command) => command.startsWith(prefix));
	if (!matches.length) return null;

	let completion: string = matches[0];
	for (const candidate of matches.slice(1)) {
		let i = 0;
		while (i < completion.length && completion[i] === candidate[i]) i += 1;
		completion = completion.slice(0, i);
	}
	if (completion.length <= prefix.length && matches.length !== 1) return null;

	const replacement = completion + (RAG_COMMANDS.includes(completion as (typeof RAG_COMMANDS)[number]) ? ' ' : '');
	return beforeCursor.slice(0, beforeCursor.length - match[2].length) + replacement + afterCursor;
}

export function suggestRagCommand(input: string, cursorPosition = input.length): string | null {
	const completed = completeRagCommand(input, cursorPosition);
	if (!completed || completed === input) return null;
	return completed.trim();
}
