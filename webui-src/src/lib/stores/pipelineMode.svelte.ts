/**
 * Stores the user-selected pipeline mode ("fast" | "balanced" | "complex").
 * Read by chatStore.getApiOptions() and sent to the backend with every request.
 */

export type PipelineMode = 'fast' | 'balanced' | 'complex';

let _mode = $state<PipelineMode>('balanced');

export function pipelineMode(): PipelineMode {
	return _mode;
}

export function setPipelineMode(mode: PipelineMode): void {
	_mode = mode;
}
