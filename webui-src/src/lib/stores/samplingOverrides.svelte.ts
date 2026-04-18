/**
 * Per-mode user sampling overrides, persisted to localStorage.
 * Only stores keys the user has explicitly changed from admin defaults.
 * Maps to Ollama key names (max_tokens → num_predict handled externally).
 */
import { browser } from '$app/environment';

const STORAGE_KEY = 'aiiab_sampling_overrides_v1';

type ModeOverrides = Record<string, number>;
type AllOverrides = Record<string, ModeOverrides>;

const EMPTY: AllOverrides = { fast: {}, balanced: {}, complex: {} };

class SamplingOverridesStore {
	private _overrides = $state<AllOverrides>({ ...EMPTY });

	constructor() {
		if (browser) {
			try {
				const stored = localStorage.getItem(STORAGE_KEY);
				if (stored) this._overrides = { ...EMPTY, ...JSON.parse(stored) };
			} catch {
				// ignore corrupt storage
			}
		}
	}

	private _persist() {
		if (browser) localStorage.setItem(STORAGE_KEY, JSON.stringify(this._overrides));
	}

	private _modeKey(mode: string): string {
		return mode === 'chat' ? 'balanced' : mode;
	}

	get(mode: string, yamlKey: string): number | undefined {
		return this._overrides[this._modeKey(mode)]?.[yamlKey];
	}

	set(mode: string, yamlKey: string, value: number) {
		const key = this._modeKey(mode);
		this._overrides = {
			...this._overrides,
			[key]: { ...this._overrides[key], [yamlKey]: value }
		};
		this._persist();
	}

	delete(mode: string, yamlKey: string) {
		const key = this._modeKey(mode);
		const next = { ...this._overrides[key] };
		delete next[yamlKey];
		this._overrides = { ...this._overrides, [key]: next };
		this._persist();
	}

	resetKeys(mode: string, yamlKeys: string[]) {
		const key = this._modeKey(mode);
		const next = { ...this._overrides[key] };
		for (const k of yamlKeys) delete next[k];
		this._overrides = { ...this._overrides, [key]: next };
		this._persist();
	}

	resetAll(mode: string) {
		const key = this._modeKey(mode);
		this._overrides = { ...this._overrides, [key]: {} };
		this._persist();
	}

	getForMode(mode: string): ModeOverrides {
		return this._overrides[this._modeKey(mode)] ?? {};
	}

	hasOverride(mode: string, yamlKey: string): boolean {
		return yamlKey in (this._overrides[this._modeKey(mode)] ?? {});
	}
}

export const samplingOverridesStore = new SamplingOverridesStore();
