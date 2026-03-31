<script lang="ts">
	import { onMount } from 'svelte';
	import { Package, Download, Trash2, Cpu, Database, RefreshCw, AlertTriangle } from '@lucide/svelte';

	// ── Types ──────────────────────────────────────────────────────────────────

	type InstalledModel = {
		name: string;
		size_bytes: number;
		size_label: string;
		param_size: string;
		quantization: string;
		family: string;
		is_embedding: boolean;
		modified_at: string;
	};

	type RegistryModel = {
		name: string;
		description?: string;
		pulls?: number;
		popular_tags?: string[];
		tags?: string[];
		tags_count?: number;
		last_updated?: string;
		namespace?: string;
	};

	type ActiveModels = {
		llm_model: string;
		embedding_model: string;
	};

	// ── State ─────────────────────────────────────────────────────────────────

	let installed: InstalledModel[] = $state([]);
	let registry: RegistryModel[] = $state([]);
	let active: ActiveModels = $state({ llm_model: '', embedding_model: '' });

	let offline = $state(false);
	let loading = $state(true);
	let error = $state('');

	let pulling: string | null = $state(null);        // model name being pulled
	let pullStatus = $state('');                       // progress text
	let deleting: string | null = $state(null);

	let restarting = $state(false);

	// Embedder warning modal
	let embedderWarning = $state(false);
	let pendingEmbedder = $state('');

	// Only allow actions from the Pi itself
	const isLocalhost =
		typeof window !== 'undefined' &&
		(window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1');

	// ── Data loading ──────────────────────────────────────────────────────────

	async function loadAll() {
		loading = true;
		error = '';
		try {
			const [instRes, regRes, actRes] = await Promise.all([
				fetch('/api/models/installed'),
				fetch('/api/models/registry'),
				fetch('/api/models/active')
			]);
			const instData = await instRes.json();
			const regData = await regRes.json();
			const actData = await actRes.json();

			installed = instData.models ?? [];
			registry = regData.models ?? [];
			offline = regData.offline ?? false;
			active = actData;
		} catch (e: any) {
			error = String(e);
		} finally {
			loading = false;
		}
	}

	onMount(loadAll);

	// ── Helpers ───────────────────────────────────────────────────────────────

	function installedNames(): Set<string> {
		// Strip tag for base-name matching (e.g. "llama3.2" matches "llama3.2:latest")
		const s = new Set<string>();
		for (const m of installed) {
			s.add(m.name);
			s.add(m.name.split(':')[0]);
		}
		return s;
	}

	function modelTags(m: RegistryModel): string[] {
		return (m.popular_tags ?? m.tags ?? []).slice(0, 5);
	}

	function formatPulls(n?: number): string {
		if (!n) return '';
		if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M pulls`;
		if (n >= 1_000) return `${(n / 1_000).toFixed(0)}K pulls`;
		return `${n} pulls`;
	}

	// ── Actions ───────────────────────────────────────────────────────────────

	async function setActiveLLM(name: string) {
		if (restarting) return;
		restarting = true;
		await fetch('/api/models/set-active', {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({ type: 'llm', model: name })
		});
	}

	function handleSetEmbedder(name: string) {
		if (name === active.embedding_model) return;
		pendingEmbedder = name;
		embedderWarning = true;
	}

	async function confirmEmbedder() {
		embedderWarning = false;
		if (restarting) return;
		restarting = true;
		await fetch('/api/models/set-active', {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({ type: 'embedder', model: pendingEmbedder, wipe_db: true })
		});
	}

	async function pullModel(name: string) {
		if (!isLocalhost || pulling) return;
		pulling = name;
		pullStatus = 'Starting…';

		const res = await fetch('/api/models/pull', {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({ name })
		});

		const reader = res.body!.getReader();
		const decoder = new TextDecoder();

		while (true) {
			const { done, value } = await reader.read();
			if (done) break;
			const text = decoder.decode(value, { stream: true });
			for (const line of text.split('\n')) {
				if (!line.startsWith('data: ')) continue;
				const raw = line.slice(6).trim();
				if (raw === '[DONE]') break;
				try {
					const msg = JSON.parse(raw);
					if (msg.error) {
						pullStatus = `Error: ${msg.error}`;
					} else if (msg.total && msg.completed) {
						const pct = Math.round((msg.completed / msg.total) * 100);
						pullStatus = `${msg.status} ${pct}%`;
					} else if (msg.status) {
						pullStatus = msg.status;
					}
				} catch {
					// ignore malformed lines
				}
			}
		}

		pulling = null;
		pullStatus = '';
		await loadAll();
	}

	async function deleteModel(name: string) {
		if (!isLocalhost || deleting) return;
		deleting = name;
		await fetch(`/api/models/${encodeURIComponent(name)}`, { method: 'DELETE' });
		deleting = null;
		await loadAll();
	}
</script>

<div class="page">
	<!-- Header -->
	<div class="header">
		<a href="/" class="back">← Chat</a>
		<h1>
			<Package class="h-5 w-5" />
			Models
		</h1>
		<button class="refresh-btn" onclick={loadAll} disabled={loading}>
			<RefreshCw class={loading ? 'h-3.5 w-3.5 spinning' : 'h-3.5 w-3.5'} />
			Refresh
		</button>
	</div>

	<!-- Offline banner -->
	{#if offline}
		<div class="offline-banner">Offline — showing installed models only</div>
	{/if}

	<!-- Restarting overlay -->
	{#if restarting}
		<div class="restart-overlay">
			<div class="restart-box">
				<div class="restart-spinner"></div>
				<div>Server restarting with new configuration…</div>
				<div class="restart-sub">Page will reload automatically.</div>
			</div>
		</div>
		<!-- Auto-reload after restart -->
		{#if restarting}
			{@html '<script>setTimeout(() => location.reload(), 4000)<\/script>'}
		{/if}
	{/if}

	{#if loading && !installed.length}
		<div class="loading">Loading…</div>
	{:else if error}
		<div class="error-msg">{error}</div>
	{:else}
		<!-- ── Active Models ──────────────────────────────────────────────── -->
		<section class="section">
			<div class="section-title">Active Configuration</div>
			<div class="active-grid">
				<div class="active-card">
					<Cpu style="color:var(--muted-foreground);width:18px;height:18px;flex-shrink:0" />
					<div>
						<div class="active-label">LLM</div>
						<div class="active-value">{active.llm_model || '—'}</div>
					</div>
				</div>
				<div class="active-card">
					<Database style="color:var(--muted-foreground);width:18px;height:18px;flex-shrink:0" />
					<div>
						<div class="active-label">Embedder</div>
						<div class="active-value">{active.embedding_model || '—'}</div>
					</div>
				</div>
			</div>
		</section>

		<!-- ── Installed Models ───────────────────────────────────────────── -->
		<section class="section">
			<div class="section-title">Installed ({installed.length})</div>
			{#if installed.length === 0}
				<div class="empty">No models installed in Ollama.</div>
			{:else}
				<div class="model-grid">
					{#each installed as m}
						<div class="model-card" class:active-llm={m.name === active.llm_model}
							 class:active-embed={m.name === active.embedding_model}>
							<div class="model-name">{m.name}</div>

							<!-- Metadata chips -->
							<div class="chips">
								{#if m.param_size}
									<span class="inline-flex items-center rounded-sm bg-muted-foreground/10 px-1.5 py-1 text-xs text-muted-foreground">
										{m.param_size}
									</span>
								{/if}
								{#if m.quantization}
									<span class="inline-flex items-center rounded-sm bg-muted-foreground/10 px-1.5 py-1 text-xs text-muted-foreground">
										{m.quantization}
									</span>
								{/if}
								{#if m.size_label}
									<span class="inline-flex items-center rounded-sm bg-muted-foreground/10 px-1.5 py-1 text-xs text-muted-foreground">
										{m.size_label}
									</span>
								{/if}
								{#if m.is_embedding}
									<span class="inline-flex items-center rounded-sm bg-blue-500/15 px-1.5 py-1 text-xs text-blue-400">
										embed
									</span>
								{/if}
								{#if m.name === active.llm_model}
									<span class="inline-flex items-center rounded-sm bg-green-500/15 px-1.5 py-1 text-xs text-green-400">
										active LLM
									</span>
								{/if}
								{#if m.name === active.embedding_model}
									<span class="inline-flex items-center rounded-sm bg-purple-500/15 px-1.5 py-1 text-xs text-purple-400">
										active embedder
									</span>
								{/if}
							</div>

							<!-- Actions (localhost only) -->
							{#if isLocalhost}
								<div class="actions">
									{#if !m.is_embedding}
										<button
											class="inline-flex cursor-pointer items-center gap-1.5 rounded-sm bg-muted-foreground/10 px-1.5 py-1 text-xs transition hover:text-foreground focus:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-60 text-foreground"
											onclick={() => setActiveLLM(m.name)}
											disabled={m.name === active.llm_model || restarting}
										>
											<Cpu class="h-3 w-3" />
											{m.name === active.llm_model ? 'Active LLM' : 'Set as LLM'}
										</button>
									{/if}
									<button
										class="inline-flex cursor-pointer items-center gap-1.5 rounded-sm bg-muted-foreground/10 px-1.5 py-1 text-xs transition hover:text-foreground focus:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-60 text-foreground"
										onclick={() => handleSetEmbedder(m.name)}
										disabled={m.name === active.embedding_model || restarting}
									>
										<Database class="h-3 w-3" />
										{m.name === active.embedding_model ? 'Active Embedder' : 'Set as Embedder'}
									</button>
									<button
										class="inline-flex cursor-pointer items-center gap-1.5 rounded-sm bg-red-500/10 px-1.5 py-1 text-xs text-red-400 transition hover:bg-red-500/20 focus:outline-none disabled:cursor-not-allowed disabled:opacity-60"
										onclick={() => deleteModel(m.name)}
										disabled={!!deleting || restarting}
									>
										<Trash2 class="h-3 w-3" />
										{deleting === m.name ? 'Deleting…' : 'Delete'}
									</button>
								</div>
							{/if}
						</div>
					{/each}
				</div>
			{/if}
		</section>

		<!-- ── Registry (online only) ─────────────────────────────────────── -->
		{#if !offline}
			<section class="section">
				<div class="section-title">Available from Registry ({registry.length})</div>
				{#if registry.length === 0}
					<div class="empty">No registry data available.</div>
				{:else}
					<div class="model-grid">
						{#each registry as m}
							{@const names = installedNames()}
							{@const alreadyInstalled = names.has(m.name) || names.has(m.name.split(':')[0])}
							{#if !alreadyInstalled}
								<div class="model-card registry-card">
									<div class="model-name">{m.name}</div>
									{#if m.description}
										<div class="model-desc">{m.description}</div>
									{/if}

									<!-- Metadata chips: popular tags -->
									<div class="chips">
										{#each modelTags(m) as tag}
											<span class="inline-flex items-center rounded-sm bg-muted-foreground/10 px-1.5 py-1 text-xs text-muted-foreground">
												{tag}
											</span>
										{/each}
										{#if m.pulls}
											<span class="inline-flex items-center rounded-sm bg-muted-foreground/10 px-1.5 py-1 text-xs text-muted-foreground">
												{formatPulls(m.pulls)}
											</span>
										{/if}
									</div>

									<!-- Install (localhost only) -->
									{#if isLocalhost}
										<div class="actions">
											{#if modelTags(m).length > 0}
												{#each modelTags(m).slice(0, 3) as tag}
													{@const fullName = `${m.name}:${tag}`}
													<button
														class="inline-flex cursor-pointer items-center gap-1.5 rounded-sm bg-muted-foreground/10 px-1.5 py-1 text-xs transition hover:text-foreground focus:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-60 text-foreground"
														onclick={() => pullModel(fullName)}
														disabled={!!pulling || restarting}
													>
														<Download class="h-3 w-3" />
														{pulling === fullName ? (pullStatus || 'Pulling…') : `Install ${tag}`}
													</button>
												{/each}
											{:else}
												<button
													class="inline-flex cursor-pointer items-center gap-1.5 rounded-sm bg-muted-foreground/10 px-1.5 py-1 text-xs transition hover:text-foreground focus:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-60 text-foreground"
													onclick={() => pullModel(m.name)}
													disabled={!!pulling || restarting}
												>
													<Download class="h-3 w-3" />
													{pulling === m.name ? (pullStatus || 'Pulling…') : 'Install'}
												</button>
											{/if}
										</div>
									{/if}
								</div>
							{/if}
						{/each}
					</div>
				{/if}
			</section>
		{/if}
	{/if}
</div>

<!-- ── Embedder change warning modal ─────────────────────────────────────── -->
{#if embedderWarning}
	<div class="modal-overlay" role="dialog" aria-modal="true">
		<div class="modal">
			<div class="modal-icon">
				<AlertTriangle class="h-6 w-6 text-yellow-400" />
			</div>
			<div class="modal-title">Change Embedding Model?</div>
			<p class="modal-body">
				Switching to <strong>{pendingEmbedder}</strong> uses a different vector space.
				The knowledge graph and query cache (built with the current embedder) will be
				<strong>wiped</strong> and rebuilt from scratch as new queries arrive.
			</p>
			<div class="modal-actions">
				<button class="modal-btn cancel" onclick={() => (embedderWarning = false)}>Cancel</button>
				<button class="modal-btn danger" onclick={confirmEmbedder}>Wipe DB &amp; Switch</button>
			</div>
		</div>
	</div>
{/if}

<style>
	.page {
		min-height: 100vh;
		background: var(--background);
		color: var(--foreground);
		font-family: 'JetBrains Mono', 'Fira Code', monospace, sans-serif;
		padding: 24px;
	}

	/* Header */
	.header {
		display: flex;
		align-items: center;
		gap: 12px;
		margin-bottom: 24px;
	}
	h1 {
		display: flex;
		align-items: center;
		gap: 8px;
		font-size: 18px;
		font-weight: 600;
		color: var(--foreground);
		margin: 0;
	}
	.back {
		color: var(--muted-foreground);
		text-decoration: none;
		font-size: 13px;
		padding: 5px 10px;
		border: 1px solid var(--border);
		border-radius: 6px;
	}
	.back:hover {
		color: var(--foreground);
		border-color: var(--ring);
	}
	.refresh-btn {
		margin-left: auto;
		display: flex;
		align-items: center;
		gap: 5px;
		background: var(--card);
		border: 1px solid var(--border);
		border-radius: 6px;
		color: var(--muted-foreground);
		font-size: 12px;
		padding: 5px 10px;
		cursor: pointer;
	}
	.refresh-btn:hover {
		color: var(--foreground);
		border-color: var(--ring);
	}
	.refresh-btn:disabled {
		opacity: 0.5;
		cursor: not-allowed;
	}

	/* Offline banner */
	.offline-banner {
		background: rgba(255, 170, 51, 0.1);
		border: 1px solid rgba(255, 170, 51, 0.3);
		border-radius: 6px;
		color: #ffaa33;
		font-size: 12px;
		padding: 8px 14px;
		margin-bottom: 20px;
	}

	/* Restart overlay */
	.restart-overlay {
		position: fixed;
		inset: 0;
		background: rgba(0, 0, 0, 0.7);
		display: flex;
		align-items: center;
		justify-content: center;
		z-index: 50;
	}
	.restart-box {
		background: var(--card);
		border: 1px solid var(--border);
		border-radius: 10px;
		padding: 32px 40px;
		text-align: center;
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: 12px;
		font-size: 14px;
		color: var(--foreground);
	}
	.restart-sub {
		font-size: 12px;
		color: var(--muted-foreground);
	}
	.restart-spinner {
		width: 28px;
		height: 28px;
		border: 3px solid var(--border);
		border-top-color: #44cc88;
		border-radius: 50%;
		animation: spin 0.8s linear infinite;
	}

	/* Sections */
	.section {
		margin-bottom: 32px;
	}
	.section-title {
		font-size: 10px;
		text-transform: uppercase;
		letter-spacing: 0.1em;
		color: var(--muted-foreground);
		margin-bottom: 12px;
	}
	.loading,
	.empty {
		color: var(--muted-foreground);
		font-size: 13px;
		padding: 20px 0;
	}
	.error-msg {
		color: #ff6666;
		font-size: 13px;
		padding: 20px 0;
	}

	/* Active models */
	.active-grid {
		display: flex;
		gap: 12px;
		flex-wrap: wrap;
	}
	.active-card {
		display: flex;
		align-items: center;
		gap: 12px;
		background: var(--card);
		border: 1px solid var(--border);
		border-radius: 8px;
		padding: 12px 16px;
		min-width: 220px;
	}
.active-label {
		font-size: 10px;
		text-transform: uppercase;
		letter-spacing: 0.08em;
		color: var(--muted-foreground);
		margin-bottom: 2px;
	}
	.active-value {
		font-size: 13px;
		color: var(--foreground);
		font-weight: 500;
	}

	/* Model cards grid */
	.model-grid {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
		gap: 12px;
	}
	.model-card {
		background: var(--card);
		border: 1px solid var(--border);
		border-radius: 8px;
		padding: 14px 16px;
		display: flex;
		flex-direction: column;
		gap: 8px;
	}
	.model-card.active-llm {
		border-color: rgba(68, 204, 136, 0.4);
	}
	.model-card.active-embed {
		border-color: rgba(168, 85, 247, 0.4);
	}
	.model-name {
		font-size: 13px;
		font-weight: 600;
		color: var(--foreground);
		word-break: break-all;
	}
	.model-desc {
		font-size: 11px;
		color: var(--muted-foreground);
		line-height: 1.4;
		display: -webkit-box;
		-webkit-line-clamp: 2;
		-webkit-box-orient: vertical;
		overflow: hidden;
	}
	.chips {
		display: flex;
		flex-wrap: wrap;
		gap: 4px;
	}
	.actions {
		display: flex;
		flex-wrap: wrap;
		gap: 4px;
		margin-top: 4px;
	}

	/* Modal */
	.modal-overlay {
		position: fixed;
		inset: 0;
		background: rgba(0, 0, 0, 0.6);
		display: flex;
		align-items: center;
		justify-content: center;
		z-index: 40;
	}
	.modal {
		background: var(--card);
		border: 1px solid var(--border);
		border-radius: 10px;
		padding: 28px 32px;
		max-width: 420px;
		width: 90%;
		display: flex;
		flex-direction: column;
		gap: 12px;
	}
	.modal-icon {
		display: flex;
	}
	.modal-title {
		font-size: 16px;
		font-weight: 600;
		color: var(--foreground);
	}
	.modal-body {
		font-size: 13px;
		color: var(--muted-foreground);
		line-height: 1.5;
		margin: 0;
	}
	.modal-body strong {
		color: var(--foreground);
	}
	.modal-actions {
		display: flex;
		gap: 8px;
		justify-content: flex-end;
		margin-top: 4px;
	}
	.modal-btn {
		font-size: 13px;
		padding: 7px 16px;
		border-radius: 6px;
		cursor: pointer;
		border: 1px solid var(--border);
		background: var(--card);
		color: var(--foreground);
	}
	.modal-btn:hover {
		border-color: var(--ring);
	}
	.modal-btn.danger {
		background: rgba(239, 68, 68, 0.15);
		border-color: rgba(239, 68, 68, 0.4);
		color: #f87171;
	}
	.modal-btn.danger:hover {
		background: rgba(239, 68, 68, 0.25);
	}

	/* Animations */
	@keyframes spin {
		to { transform: rotate(360deg); }
	}
	:global(.spinning) {
		animation: spin 0.8s linear infinite;
	}
</style>
