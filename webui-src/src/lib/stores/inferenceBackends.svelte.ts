/**
 * inferenceBackends — manages the inference backend priority chain.
 *
 * Backends are Ollama-compatible servers that can serve LLM requests.
 * Priority order (lower number = tried first):
 *   - Android phone MLC (auto-discovered via mDNS, priority 1) — builtin, managed by server
 *   - User backends (configurable in Settings, priority 10+)
 *   - Pi Local Ollama (always last fallback, priority 999) — builtin
 *
 * The active backend name is read from SSE finish chunks (delta.backend).
 */

export interface InferenceBackend {
	id: string;
	name: string;
	url: string;
	priority: number;
	builtin: boolean;
	enabled: boolean;
	healthy?: boolean;
}

class InferenceBackendsStore {
	backends = $state<InferenceBackend[]>([]);
	activeBackendName = $state<string | null>(null);
	loading = $state(false);
	saving = $state(false);

	/** Derived: the name of the top-priority healthy backend (from last fetch). */
	get topPriorityName(): string | null {
		const sorted = [...this.backends]
			.filter((b) => b.enabled && b.url && b.healthy)
			.sort((a, b) => a.priority - b.priority);
		return sorted[0]?.name ?? null;
	}

	/** True when active backend is NOT the top-priority one (degraded mode). */
	get isDegraded(): boolean {
		if (!this.activeBackendName) return false;
		const top = this.topPriorityName;
		if (!top) return false;
		return this.activeBackendName !== top;
	}

	/** The Pi Local backend for reference. */
	get piLocal(): InferenceBackend | undefined {
		return this.backends.find((b) => b.id === 'pi-local');
	}

	/** User-configurable (non-builtin) backends. */
	get userBackends(): InferenceBackend[] {
		return this.backends.filter((b) => !b.builtin);
	}

	async load() {
		this.loading = true;
		try {
			const res = await fetch('/api/backends');
			if (res.ok) {
				const data = await res.json();
				this.backends = data.backends ?? [];
			}
		} catch (e) {
			console.error('[inferenceBackends] load failed', e);
		} finally {
			this.loading = false;
		}
	}

	async save(userBackends: InferenceBackend[]) {
		this.saving = true;
		try {
			const res = await fetch('/api/backends', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ backends: userBackends })
			});
			if (res.ok) {
				await this.load();
			}
		} catch (e) {
			console.error('[inferenceBackends] save failed', e);
		} finally {
			this.saving = false;
		}
	}

	async testUrl(url: string): Promise<{ healthy: boolean; models: string[] }> {
		try {
			const res = await fetch(`/api/backends/health?url=${encodeURIComponent(url)}`);
			if (res.ok) return await res.json();
		} catch {
			/* fall through */
		}
		return { healthy: false, models: [] };
	}

	/** Called by chat stream handler when a finish SSE chunk arrives with backend name. */
	setActiveBackend(name: string | null) {
		this.activeBackendName = name;
	}
}

export const inferenceBackendsStore = new InferenceBackendsStore();
