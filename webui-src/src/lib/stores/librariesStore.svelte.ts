export interface ZimLibrary {
	name: string;
	title: string;
	enabled: boolean;
	count: number;
	locked: boolean;
	is_indexed: boolean;
	zim_size_mb: number;
	db_size_mb: number;
	index_size_mb: number;
	vector_count: number;
	chunk_count: number;
	indexed_article_count: number;
	archive_article_count: number;
	// On-demand / title index
	title_index_enabled: boolean;
	title_index_ready: boolean;
	title_index_count: number;
	max_articles: number;
	max_vectors: number;
}

export interface TitleIndexStatus {
	ready: boolean;
	count: number;
	building: boolean;
	finished: boolean;
	error: string | null;
	progress: { done: number; total: number; pct: number };
}

export interface EmbedLimits {
	embedded_articles: number;
	max_articles: number;
	embedded_vectors: number;
	max_vectors: number;
	articles_pct: number;
	vectors_pct: number;
}

function createLibrariesStore() {
	let libraries = $state<ZimLibrary[]>([]);
	let loading = $state(false);
	let rescanning = $state(false);
	let error = $state<string | null>(null);

	// Per-ZIM title index build state
	let titleIndexStatus = $state<Record<string, TitleIndexStatus>>({});
	let embedLimits = $state<Record<string, EmbedLimits>>({});

	async function load() {
		loading = true;
		error = null;
		try {
			const res = await fetch('/api/libraries');
			const data = await res.json();
			libraries = data.libraries ?? [];
		} catch (e) {
			error = 'Failed to load libraries';
		} finally {
			loading = false;
		}
	}

	async function rescan() {
		rescanning = true;
		error = null;
		try {
			const res = await fetch('/api/libraries/rescan', { method: 'POST' });
			const data = await res.json();
			libraries = data.libraries ?? [];
		} catch (e) {
			error = 'Rescan failed';
		} finally {
			rescanning = false;
		}
	}

	async function update(name: string, patch: { enabled?: boolean; count?: number }) {
		try {
			const res = await fetch(`/api/libraries/${encodeURIComponent(name)}`, {
				method: 'PUT',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify(patch)
			});
			if (!res.ok) return false;
			libraries = libraries.map((lib) => (lib.name === name ? { ...lib, ...patch } : lib));
			return true;
		} catch {
			return false;
		}
	}

	async function buildTitleIndex(name: string) {
		titleIndexStatus = {
			...titleIndexStatus,
			[name]: { ready: false, count: 0, building: true, finished: false, error: null,
			           progress: { done: 0, total: 0, pct: 0 } }
		};
		try {
			await fetch(`/api/libraries/${encodeURIComponent(name)}/build-title-index`, { method: 'POST' });
			pollTitleIndexStatus(name);
		} catch (e) {
			titleIndexStatus = {
				...titleIndexStatus,
				[name]: { ...titleIndexStatus[name], building: false, error: 'Failed to start' }
			};
		}
	}

	function pollTitleIndexStatus(name: string) {
		const interval = setInterval(async () => {
			try {
				const res = await fetch(`/api/libraries/${encodeURIComponent(name)}/title-index-status`);
				const data = await res.json();
				titleIndexStatus = { ...titleIndexStatus, [name]: data };
				if (!data.building) {
					clearInterval(interval);
					if (data.finished && !data.error) load(); // refresh library list
				}
			} catch {
				clearInterval(interval);
			}
		}, 2000);
	}

	async function fetchTitleIndexStatus(name: string) {
		try {
			const res = await fetch(`/api/libraries/${encodeURIComponent(name)}/title-index-status`);
			const data = await res.json();
			titleIndexStatus = { ...titleIndexStatus, [name]: data };
			if (data.building) pollTitleIndexStatus(name);
		} catch { /* ignore */ }
	}

	async function embedArticle(name: string, path: string): Promise<{ ok: boolean; error?: string; chunks_added?: number }> {
		try {
			const res = await fetch(`/api/embed/${encodeURIComponent(name)}`, {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ path })
			});
			return await res.json();
		} catch {
			return { ok: false, error: 'Request failed' };
		}
	}

	async function fetchEmbedLimits(name: string) {
		try {
			const res = await fetch(`/api/libraries/${encodeURIComponent(name)}/embed-limits`);
			const data = await res.json();
			if (data.ok) {
				const { ok: _, ...limits } = data;
				embedLimits = { ...embedLimits, [name]: limits as EmbedLimits };
			}
		} catch { /* ignore */ }
	}

	return {
		get libraries() { return libraries; },
		get loading() { return loading; },
		get rescanning() { return rescanning; },
		get error() { return error; },
		get titleIndexStatus() { return titleIndexStatus; },
		get embedLimits() { return embedLimits; },
		load,
		rescan,
		update,
		buildTitleIndex,
		fetchTitleIndexStatus,
		embedArticle,
		fetchEmbedLimits,
	};
}

export const librariesStore = createLibrariesStore();
