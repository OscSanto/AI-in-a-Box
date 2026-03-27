<script lang="ts">
	import { onMount, onDestroy } from 'svelte';

	let container: HTMLDivElement;
	let graphInstance: any;

	let stats = { articles: 0, edges: 0, queries: 0 };
	let searchTerm = '';
	let loading = true;
	let error = '';

	// Relation → color mapping
	const RELATION_COLORS: Record<string, string> = {
		antagonist: '#ff4455',
		villain: '#ff4455',
		'crime lords': '#ff6622',
		'rogues gallery': '#ff6622',
		supporting: '#44aaff',
		ally: '#44ff88',
		allies: '#44ff88',
		family: '#aa44ff',
		'alternative versions': '#ffaa44',
		location: '#44ffdd',
		locations: '#44ffdd',
		index: '#ffee44',
		indexes: '#ffee44',
		securities: '#88ffcc',
		futures: '#ffcc44',
		publication: '#cc88ff',
		film: '#ff88cc'
	};

	function relationColor(relation: string): string {
		const r = (relation || '').toLowerCase();
		for (const [key, color] of Object.entries(RELATION_COLORS)) {
			if (r.includes(key)) return color;
		}
		return '#556677';
	}

	async function fetchGraph(center?: string, hops = 2) {
		const params = new URLSearchParams({ hops: String(hops) });
		if (center) params.set('center', center);
		const res = await fetch(`/visual/api/graph?${params}`);
		return res.json();
	}

	async function fetchStats() {
		const res = await fetch('/visual/api/stats');
		stats = await res.json();
	}

	async function expandNode(node: any) {
		const data = await fetchGraph(node.id, 1);
		const current = graphInstance.graphData();
		const existingIds = new Set(current.nodes.map((n: any) => n.id));
		graphInstance.graphData({
			nodes: [...current.nodes, ...data.nodes.filter((n: any) => !existingIds.has(n.id))],
			links: [...current.links, ...data.links]
		});
	}

	function handleSearch(e: KeyboardEvent) {
		if (e.key === 'Enter' && searchTerm.trim()) {
			fetchGraph(searchTerm.trim(), 2).then((data) => graphInstance?.graphData(data));
		}
	}

	onMount(async () => {
		try {
			const [{ default: ForceGraph3D }, { default: ForceGraph2D }, initialData] = await Promise.all([
				import('3d-force-graph'),
				import('force-graph'),
				fetchGraph()
			]);
			await fetchStats();
			loading = false;

			function applyCommon(g: any, is3d: boolean) {
				g.graphData(initialData)
					.nodeLabel('id')
					.nodeVal((n: any) => Math.max(1, Math.sqrt((n.times_selected || 0) + 1) * 4))
					.nodeColor((n: any) => {
						if (n.ghost) return '#1e2a3a';
						const s = n.times_selected || 0;
						if (s > 10) return '#ffffff';
						if (s > 4) return '#aaccff';
						if (s > 1) return '#6688bb';
						return '#334455';
					})
					.linkLabel((l: any) => (l.navbox ? `${l.relation} — ${l.navbox}` : l.relation))
					.linkColor((l: any) => relationColor(l.relation))
					.linkWidth(is3d ? 0.4 : 1)
					.backgroundColor('#070d14')
					.onNodeClick((node: any) => expandNode(node.id))
					.width(window.innerWidth)
					.height(window.innerHeight);
				if (is3d) g.nodeOpacity(0.9).linkOpacity(0.35);
				else g.linkDirectionalArrowLength(4).linkDirectionalArrowRelPos(1);
				return g;
			}

			try {
				graphInstance = applyCommon(ForceGraph3D()(container), true);
			} catch {
				console.warn('[visual] WebGL unavailable, falling back to 2D');
				container.innerHTML = '';
				graphInstance = applyCommon(ForceGraph2D()(container), false);
			}
		} catch (e: any) {
			error = String(e);
			loading = false;
			console.error('[visual] init failed:', e);
		}
	});

	onDestroy(() => {
		graphInstance?._destructor?.();
	});

	const LEGEND_ENTRIES = Object.entries(RELATION_COLORS).slice(0, 10);
</script>

<div bind:this={container} class="graph-container"></div>

{#if loading}
	<div class="loading">Loading graph…</div>
{/if}

{#if error}
	<div class="error">{error}</div>
{/if}

<div class="overlay">
	<a href="/" class="back-btn">← Chat</a>
	<div class="stats">
		<span>{stats.articles} articles</span>
		<span>{stats.edges} edges</span>
		<span>{stats.queries} queries</span>
	</div>
	<input
		bind:value={searchTerm}
		onkeydown={handleSearch}
		placeholder="Center on article…"
		class="search-input"
	/>
</div>

<div class="legend">
	<div class="legend-title">Relations</div>
	{#each LEGEND_ENTRIES as [rel, color]}
		<div class="legend-item">
			<span class="dot" style="background:{color}"></span>
			<span>{rel}</span>
		</div>
	{/each}
</div>

<style>
	.graph-container {
		position: fixed;
		inset: 0;
		background: #070d14;
	}

	.error {
		position: fixed;
		inset: 0;
		display: flex;
		align-items: center;
		justify-content: center;
		color: #ff6666;
		font-family: monospace;
		font-size: 13px;
		background: #070d14;
		padding: 2rem;
		white-space: pre-wrap;
		z-index: 30;
	}

	.loading {
		position: fixed;
		inset: 0;
		display: flex;
		align-items: center;
		justify-content: center;
		color: #aaa;
		font-family: sans-serif;
		font-size: 14px;
		background: #070d14;
		z-index: 20;
	}

	.overlay {
		position: fixed;
		top: 16px;
		left: 16px;
		z-index: 10;
		display: flex;
		flex-direction: column;
		gap: 8px;
	}

	.back-btn {
		color: var(--card-foreground);
		text-decoration: none;
		background: var(--card);
		padding: 6px 12px;
		border-radius: 6px;
		font-size: 13px;
		font-family: sans-serif;
		border: 1px solid var(--border);
	}

	.back-btn:hover {
		color: var(--foreground);
		border-color: var(--ring);
	}

	.stats {
		background: var(--card);
		border: 1px solid var(--border);
		border-radius: 6px;
		padding: 8px 12px;
		color: var(--muted-foreground);
		font-size: 12px;
		font-family: sans-serif;
		display: flex;
		gap: 14px;
	}

	.search-input {
		background: var(--card);
		border: 1px solid var(--border);
		border-radius: 6px;
		padding: 7px 12px;
		color: var(--card-foreground);
		font-size: 13px;
		font-family: sans-serif;
		outline: none;
		width: 230px;
	}

	.search-input::placeholder {
		color: var(--muted-foreground);
	}

	.legend {
		position: fixed;
		bottom: 16px;
		right: 16px;
		z-index: 10;
		background: var(--card);
		border: 1px solid var(--border);
		border-radius: 6px;
		padding: 10px 14px;
		display: flex;
		flex-direction: column;
		gap: 5px;
		max-height: 280px;
		overflow-y: auto;
	}

	.legend-title {
		color: var(--muted-foreground);
		font-size: 10px;
		font-family: sans-serif;
		text-transform: uppercase;
		letter-spacing: 0.08em;
		margin-bottom: 2px;
	}

	.legend-item {
		display: flex;
		align-items: center;
		gap: 7px;
		color: var(--muted-foreground);
		font-size: 11px;
		font-family: sans-serif;
	}

	.dot {
		width: 8px;
		height: 8px;
		border-radius: 50%;
		flex-shrink: 0;
	}
</style>
