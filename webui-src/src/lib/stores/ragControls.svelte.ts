import { setPipelineMode, type PipelineMode } from '$lib/stores/pipelineMode.svelte';

export type RagMode = PipelineMode | 'chat';
export type RagLogLevel = 'off' | 'summary' | 'full';

export const RAG_COMMANDS = [
	'/fast',
	'/balanced',
	'/complex',
	'/chat',
	'/zim',
	'/zim all',
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

export interface RagControlsState {
	mode: RagMode;
	selectedZim: string;
	bypassCache: boolean;
	logLevel: RagLogLevel;
	thinking: boolean;
	forkActive: boolean;
}

const DEFAULT_STATE: RagControlsState = {
	mode: 'chat',
	selectedZim: 'all',
	bypassCache: false,
	logLevel: 'full',
	thinking: false,
	forkActive: false
};

let _state = $state<RagControlsState>({ ...DEFAULT_STATE });

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

export function setSelectedZim(selectedZim: string): void {
	_state = { ..._state, selectedZim: selectedZim || 'all' };
}

export function setBypassCache(bypassCache: boolean): void {
	_state = { ..._state, bypassCache };
}

export function setLogLevel(logLevel: RagLogLevel): void {
	_state = { ..._state, logLevel };
}

export function resetRagControls(): void {
	_state = { ...DEFAULT_STATE };
	syncPipelineMode(DEFAULT_STATE.mode);
}



export function resetRagMode(): void {
	setRagMode(DEFAULT_STATE.mode);
}

export function resetSelectedZim(): void {
	setSelectedZim(DEFAULT_STATE.selectedZim);
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

	if (command === 'fast' || command === 'balanced' || command === 'balance' || command === 'complex') {
		setRagMode(command === 'balance' ? 'balanced' : command);
		return { handled: true, remaining: text.slice(consumed).trimStart() };
	}
	if (command === 'chat') {
		setRagMode('chat');
		return { handled: true, remaining: text.slice(consumed).trimStart() };
	}
	if (command === 'zim') {
		if (!arg) arg = consumeNextToken();
		setSelectedZim(arg || 'all');
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
				'Commands: /fast, /balanced, /complex, /chat, /zim <name>, /zim all, /new, /cache, /log off|summary|full, /reset',
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
