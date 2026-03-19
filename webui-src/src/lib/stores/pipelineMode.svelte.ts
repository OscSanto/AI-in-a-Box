/**
 * Stores the user-selected pipeline mode ("kiwix" | "chat" | "summarize").
 * Read by chatStore.getApiOptions() and sent to the backend with every request.
 */

export type PipelineMode = 'kiwix' | 'chat' | 'summarize';

let _mode = $state<PipelineMode>('kiwix');

export function pipelineMode(): PipelineMode {
	return _mode;
}

export function setPipelineMode(mode: PipelineMode): void {
	_mode = mode;
}
