export type PipelineMode = 'balanced';

let _mode = $state<PipelineMode>('balanced');

export function pipelineMode(): PipelineMode {
	return _mode;
}

export function setPipelineMode(mode: PipelineMode): void {
	_mode = mode;
}
