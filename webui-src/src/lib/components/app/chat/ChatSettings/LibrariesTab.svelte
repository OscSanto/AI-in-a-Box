<script lang="ts">
	import { onMount } from 'svelte';
	import { RefreshCw, CheckCircle, XCircle, Database, HardDrive, Lock, BookText, Cpu } from '@lucide/svelte';
	import { librariesStore } from '$lib/stores/librariesStore.svelte';

	let embedInputs = $state<Record<string, string>>({});
	let embedResults = $state<Record<string, string>>({});
	let embedding = $state<Record<string, boolean>>({});

	onMount(() => {
		librariesStore.load();
	});

	function formatMb(mb: number): string {
		if (mb < 1) return `${Math.round(mb * 1024)} KB`;
		if (mb >= 1024) return `${(mb / 1024).toFixed(1)} GB`;
		return `${mb.toFixed(1)} MB`;
	}

	function formatCount(n: number): string {
		if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
		if (n >= 1_000) return `${(n / 1_000).toFixed(0)}K`;
		return String(n);
	}

	async function handleToggle(name: string, enabled: boolean) {
		await librariesStore.update(name, { enabled });
	}

	async function handleCountChange(name: string, raw: string) {
		const n = parseInt(raw, 10);
		if (!isNaN(n) && n >= 1) {
			await librariesStore.update(name, { count: n });
		}
	}

	async function handleBuildTitleIndex(name: string) {
		await librariesStore.buildTitleIndex(name);
	}

	async function handleEmbed(name: string) {
		const path = (embedInputs[name] ?? '').trim();
		if (!path) return;
		embedding = { ...embedding, [name]: true };
		embedResults = { ...embedResults, [name]: '' };
		const result = await librariesStore.embedArticle(name, path);
		embedding = { ...embedding, [name]: false };
		if (result.ok) {
			embedResults = { ...embedResults, [name]: result.chunks_added === 0
				? 'Already embedded.'
				: `✓ ${result.chunks_added} chunks added.` };
			embedInputs = { ...embedInputs, [name]: '' };
		} else {
			embedResults = { ...embedResults, [name]: `✗ ${result.error}` };
		}
	}

	function titleIndexLabel(name: string) {
		const s = librariesStore.titleIndexStatus[name];
		if (!s) return null;
		if (s.building) return `Building… ${s.progress.pct.toFixed(0)}%`;
		if (s.error) return `Error: ${s.error}`;
		if (s.finished) return `Done — ${formatCount(s.count)} titles`;
		return null;
	}
</script>

<div class="space-y-4">
	<!-- Header -->
	<div class="flex items-center justify-between pb-2 border-b border-border/30">
		<p class="text-xs text-muted-foreground">
			ZIM libraries discovered in the configured folder. Toggle to include/exclude from search.
		</p>
		<button
			onclick={() => librariesStore.rescan()}
			disabled={librariesStore.rescanning}
			class="flex items-center gap-1.5 rounded px-2 py-1 text-xs text-muted-foreground hover:bg-accent hover:text-foreground transition-colors disabled:opacity-50"
			title="Rescan ZIM folder for new or removed files"
		>
			<RefreshCw class="h-3 w-3 {librariesStore.rescanning ? 'animate-spin' : ''}" />
			{librariesStore.rescanning ? 'Scanning…' : 'Rescan'}
		</button>
	</div>

	{#if librariesStore.error}
		<p class="text-xs text-destructive">{librariesStore.error}</p>
	{/if}

	{#if librariesStore.loading}
		<div class="flex items-center gap-2 py-8 justify-center text-muted-foreground text-sm">
			<RefreshCw class="h-4 w-4 animate-spin" />
			Loading libraries…
		</div>
	{:else if librariesStore.libraries.length === 0}
		<div class="py-8 text-center text-sm text-muted-foreground">
			No ZIM files found. Place <code>.zim</code> files in the configured ZIM folder and click Rescan.
		</div>
	{:else}
		<div class="space-y-3">
			{#each librariesStore.libraries as lib (lib.name)}
				<div class="rounded-lg border border-border/40 bg-card p-4 space-y-3">

					<!-- Top row: toggle + name + badges -->
					<div class="flex items-start justify-between gap-3">
						<div class="flex items-center gap-3 min-w-0">
							<!-- Toggle or lock -->
							{#if lib.locked}
								<div class="flex h-5 w-9 flex-shrink-0 items-center justify-center"
									title="Managed by admin — cannot be changed">
									<Lock class="h-4 w-4 text-muted-foreground" />
								</div>
							{:else}
								<button
									role="switch"
									aria-checked={lib.enabled}
									onclick={() => handleToggle(lib.name, !lib.enabled)}
									class="relative inline-flex h-5 w-9 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors focus:outline-none {lib.enabled ? 'bg-primary' : 'bg-muted'}"
									title="{lib.enabled ? 'Disable' : 'Enable'} this library"
								>
									<span class="pointer-events-none inline-block h-4 w-4 transform rounded-full bg-white shadow ring-0 transition-transform {lib.enabled ? 'translate-x-4' : 'translate-x-0'}"></span>
								</button>
							{/if}

							<!-- Name & title -->
							<div class="min-w-0">
								<p class="text-sm font-medium truncate">{lib.title || lib.name}</p>
								<p class="text-xs text-muted-foreground truncate font-mono">{lib.name}</p>
							</div>
						</div>

						<!-- Status badges -->
						<div class="flex flex-col items-end gap-1 flex-shrink-0">
							{#if lib.is_indexed}
								<span class="flex items-center gap-1 rounded-full bg-green-100 dark:bg-green-900/30 px-2 py-0.5 text-xs text-green-700 dark:text-green-400">
									<CheckCircle class="h-3 w-3" />FAISS
								</span>
							{:else}
								<span class="flex items-center gap-1 rounded-full bg-amber-100 dark:bg-amber-900/30 px-2 py-0.5 text-xs text-amber-700 dark:text-amber-400">
									<XCircle class="h-3 w-3" />Xapian only
								</span>
							{/if}
							{#if lib.title_index_ready}
								<span class="flex items-center gap-1 rounded-full bg-blue-100 dark:bg-blue-900/30 px-2 py-0.5 text-xs text-blue-700 dark:text-blue-400">
									<BookText class="h-3 w-3" />{formatCount(lib.title_index_count)} titles
								</span>
							{/if}
						</div>
					</div>

					<!-- Stats grid -->
					<div class="grid grid-cols-2 gap-x-4 gap-y-1 text-xs text-muted-foreground">
						<div class="flex items-center gap-1.5">
							<HardDrive class="h-3 w-3 flex-shrink-0" />
							<span>ZIM: <strong class="text-foreground">{formatMb(lib.zim_size_mb)}</strong></span>
						</div>
						{#if lib.is_indexed}
							<div class="flex items-center gap-1.5">
								<Database class="h-3 w-3 flex-shrink-0" />
								<span>DB: <strong class="text-foreground">{formatMb(lib.db_size_mb)}</strong></span>
							</div>
							<div>Vectors: <strong class="text-foreground">{formatCount(lib.vector_count)}</strong> / {formatCount(lib.max_vectors)}</div>
							<div>Chunks: <strong class="text-foreground">{formatCount(lib.chunk_count)}</strong></div>
							<div>Articles (indexed): <strong class="text-foreground">{formatCount(lib.indexed_article_count)}</strong> / {formatCount(lib.max_articles)}</div>
						{/if}
						<div>Articles (total): <strong class="text-foreground">{formatCount(lib.archive_article_count)}</strong></div>
					</div>

					<!-- Controls row (count + title index + embed) -->
					<div class="space-y-2 pt-1 border-t border-border/20">

						<!-- Count input (unlocked + enabled only) -->
						{#if lib.enabled && !lib.locked}
							<div class="flex items-center gap-2">
								<label class="text-xs text-muted-foreground" for="count-{lib.name}">Results per query:</label>
								<input
									id="count-{lib.name}"
									type="number" min="1" max="50"
									value={lib.count}
									onchange={(e) => handleCountChange(lib.name, (e.currentTarget as HTMLInputElement).value)}
									class="w-16 rounded border border-border bg-background px-2 py-0.5 text-xs text-foreground focus:outline-none focus:ring-1 focus:ring-ring"
								/>
							</div>
						{:else if lib.locked}
							<div class="flex items-center gap-1.5 text-xs text-muted-foreground">
								<Lock class="h-3 w-3" />Managed by admin
							</div>
						{/if}

						<!-- Title index build (opt-in ZIMs only) -->
						{#if lib.title_index_enabled}
							{@const tiStatus = librariesStore.titleIndexStatus[lib.name]}
							<div class="flex items-center gap-2">
								<button
									onclick={() => handleBuildTitleIndex(lib.name)}
									disabled={tiStatus?.building}
									class="flex items-center gap-1.5 rounded px-2 py-1 text-xs border border-border/50 text-muted-foreground hover:bg-accent hover:text-foreground transition-colors disabled:opacity-50"
									title="Build or rebuild semantic title index"
								>
									<BookText class="h-3 w-3 {tiStatus?.building ? 'animate-pulse' : ''}" />
									{lib.title_index_ready ? 'Rebuild title index' : 'Build title index'}
								</button>
								{#if titleIndexLabel(lib.name)}
									<span class="text-xs text-muted-foreground">{titleIndexLabel(lib.name)}</span>
								{/if}
							</div>
						{/if}

						<!-- On-demand article embed (indexed ZIMs only) -->
						{#if lib.is_indexed}
							<div class="flex items-center gap-2">
								<Cpu class="h-3 w-3 flex-shrink-0 text-muted-foreground" />
								<input
									type="text"
									placeholder="Article path  e.g. A/Heart_attack"
									bind:value={embedInputs[lib.name]}
									onkeydown={(e) => e.key === 'Enter' && handleEmbed(lib.name)}
									class="flex-1 min-w-0 rounded border border-border bg-background px-2 py-0.5 text-xs text-foreground placeholder:text-muted-foreground/60 focus:outline-none focus:ring-1 focus:ring-ring"
								/>
								<button
									onclick={() => handleEmbed(lib.name)}
									disabled={embedding[lib.name] || !(embedInputs[lib.name] ?? '').trim()}
									class="flex-shrink-0 rounded px-2 py-1 text-xs border border-border/50 text-muted-foreground hover:bg-accent hover:text-foreground transition-colors disabled:opacity-50"
								>
									{embedding[lib.name] ? 'Embedding…' : 'Embed'}
								</button>
							</div>
							{#if embedResults[lib.name]}
								<p class="text-xs {embedResults[lib.name].startsWith('✓') ? 'text-green-600 dark:text-green-400' : 'text-destructive'}">
									{embedResults[lib.name]}
								</p>
							{/if}
						{/if}
					</div>

				</div>
			{/each}
		</div>
	{/if}
</div>
