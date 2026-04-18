/**
 * Fetches and caches /api/sampling-config.
 * Provides Ollama hardcoded defaults, per-mode admin llm_options, and per-mode caps.
 */

interface ModeConfig {
	llm_options: Record<string, number>;
	caps: Record<string, number>;
}

interface SamplingConfigData {
	ollama_defaults: Record<string, number>;
	modes: Record<string, ModeConfig>;
}

class SamplingConfigStore {
	data = $state<SamplingConfigData | null>(null);
	loading = $state(false);

	async load(): Promise<void> {
		if (this.data || this.loading) return;
		this.loading = true;
		try {
			const res = await fetch('/api/sampling-config');
			if (res.ok) this.data = await res.json();
		} finally {
			this.loading = false;
		}
	}

	get ollamaDefaults(): Record<string, number> {
		return this.data?.ollama_defaults ?? {};
	}

	getAdminDefaults(mode: string): Record<string, number> {
		const effectiveMode = mode === 'chat' ? 'balanced' : mode;
		return this.data?.modes[effectiveMode]?.llm_options ?? {};
	}

	getCaps(mode: string): Record<string, number> {
		const effectiveMode = mode === 'chat' ? 'balanced' : mode;
		return this.data?.modes[effectiveMode]?.caps ?? {};
	}

	/** Returns the effective display default for a UI key in a given mode.
	 *  UI key max_tokens maps to YAML key num_predict. */
	getEffectiveDefault(mode: string, uiKey: string): number | undefined {
		const yamlKey = uiKey === 'max_tokens' ? 'num_predict' : uiKey;
		const admin = this.getAdminDefaults(mode);
		if (yamlKey in admin) return admin[yamlKey];
		const ollama = this.ollamaDefaults;
		if (yamlKey in ollama) return ollama[yamlKey];
		return undefined;
	}

	getCap(mode: string, uiKey: string): number | undefined {
		const yamlKey = uiKey === 'max_tokens' ? 'num_predict' : uiKey;
		const caps = this.getCaps(mode);
		return caps[yamlKey];
	}
}

export const samplingConfigStore = new SamplingConfigStore();
