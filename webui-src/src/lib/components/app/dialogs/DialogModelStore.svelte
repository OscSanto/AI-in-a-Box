<script lang="ts">
	import { Check, Loader2, Search, AlertTriangle, Download, ExternalLink, ChevronLeft } from '@lucide/svelte';
	import * as Dialog from '$lib/components/ui/dialog';
	import { cn } from '$lib/components/ui/utils';
	import { modelsStore, modelOptions } from '$lib/stores/models.svelte';
	import { serverStore } from '$lib/stores/server.svelte';
	import { modelSwitchStore } from '$lib/stores/modelSwitch.svelte';

	interface Props {
		open?: boolean;
		onOpenChange?: (open: boolean) => void;
	}

	let { open = $bindable(), onOpenChange }: Props = $props();

	type StoreVariant  = { tag: string; params: string; description: string };
	type StoreFamily   = { name: string; description: string; variants: StoreVariant[] };
	type PullState     = { status: 'idle' | 'pulling' | 'done' | 'error'; progress: string; resolvedTag?: string };
	type HFModel       = { id: string; ollama_tag: string; downloads: number; likes: number; pipeline_tag: string };
	type GGUFFile      = { filename: string; size: number; url: string };

	let activeTab      = $state<'all' | 'installed' | 'huggingface'>('all');
	let searchQuery    = $state('');
	let customInput    = $state('');
	let catalog        = $state<StoreFamily[]>([]);
	let catalogLoading = $state(false);
	let pullStates     = $state<Record<string, PullState>>({});

	// Hugging Face
	let hfQuery    = $state('');
	let hfResults  = $state<HFModel[]>([]);
	let hfLoading  = $state(false);
	let hfError    = $state('');
	let hfSearched = $state(false);

	// llama.cpp — GGUF file picker
	let ggufRepo    = $state('');     // repo currently expanded
	let ggufFiles   = $state<GGUFFile[]>([]);
	let ggufLoading = $state(false);
	let ggufError   = $state('');

	// Download states keyed by filename
	type DlState = { status: 'idle' | 'downloading' | 'done' | 'error'; pct: number; msg: string };
	let dlStates = $state<Record<string, DlState>>({});

	let switchingTo = $derived(modelSwitchStore.switchingTo);

	let backend         = $derived((serverStore.props as any)?.backend ?? 'ollama');
	let isLlamacpp      = $derived(backend === 'llamacpp');
	let installedModels = $derived(modelOptions());
	let currentModel    = $derived(modelsStore.singleModelName);
	let installedTags   = $derived(new Set(installedModels.map((m) => m.model)));

	// ── catalog ───────────────────────────────────────────────────────────────────
	async function loadCatalog() {
		if (catalog.length > 0) return;
		catalogLoading = true;
		try {
			const res = await fetch('/api/model-store');
			catalog = (await res.json()).families ?? [];
		} catch { catalog = []; }
		finally { catalogLoading = false; }
	}

	// ── HF search ─────────────────────────────────────────────────────────────────
	async function searchHF() {
		hfLoading = true;
		hfError   = '';
		hfSearched = true;
		ggufRepo   = '';
		try {
			const res  = await fetch(`/api/hf/search?q=${encodeURIComponent(hfQuery.trim())}&limit=20`);
			const data = await res.json();
			if (data.error) { hfError = data.error; hfResults = []; }
			else             hfResults = data.models ?? [];
		} catch (e) {
			hfError   = e instanceof Error ? e.message : 'Failed to reach Hugging Face';
			hfResults = [];
		} finally { hfLoading = false; }
	}

	$effect(() => {
		if (activeTab === 'huggingface' && !hfSearched) searchHF();
	});

	// ── llamacpp: fetch GGUF files for a repo ─────────────────────────────────────
	async function openRepo(repoId: string) {
		ggufRepo    = repoId;
		ggufFiles   = [];
		ggufError   = '';
		ggufLoading = true;
		try {
			const res  = await fetch(`/api/llamacpp/repo-files?repo=${encodeURIComponent(repoId)}`);
			const data = await res.json();
			if (data.error) { ggufError = data.error; }
			else             ggufFiles = data.files ?? [];
		} catch (e) {
			ggufError = e instanceof Error ? e.message : 'Failed to fetch files';
		} finally { ggufLoading = false; }
	}

	// ── llamacpp: download a GGUF file ────────────────────────────────────────────
	async function downloadGGUF(url: string, filename: string) {
		if (dlStates[filename]?.status === 'downloading') return;
		dlStates[filename] = { status: 'downloading', pct: 0, msg: 'Starting…' };
		try {
			const res    = await fetch('/api/llamacpp/download', {
				method:  'POST',
				headers: { 'Content-Type': 'application/json' },
				body:    JSON.stringify({ url, filename }),
			});
			const reader = res.body?.getReader();
			if (!reader) throw new Error('No response body');
			const dec = new TextDecoder();
			let buf = '';
			while (true) {
				const { done, value } = await reader.read();
				if (done) break;
				buf += dec.decode(value, { stream: true });
				const lines = buf.split('\n'); buf = lines.pop() ?? '';
				for (const line of lines) {
					if (!line.startsWith('data: ')) continue;
					const raw = line.slice(6).trim();
					if (raw === '[DONE]') break;
					try {
						const msg = JSON.parse(raw);
						if (msg.error) {
							dlStates[filename] = { status: 'error', pct: 0, msg: msg.error };
							return;
						}
						if (msg.done) {
							dlStates[filename] = { status: 'done', pct: 100, msg: 'Installed' };
							await modelsStore.fetch(true);
							return;
						}
						const mb = (msg.received / 1e6).toFixed(0);
						const tot = msg.total ? `/ ${(msg.total / 1e6).toFixed(0)} MB` : '';
						dlStates[filename] = { status: 'downloading', pct: msg.pct ?? 0, msg: `${mb} ${tot} MB` };
					} catch { /* skip */ }
				}
			}
		} catch (e) {
			dlStates[filename] = { status: 'error', pct: 0, msg: e instanceof Error ? e.message : 'Failed' };
		}
	}

	// ── ollama: pull ──────────────────────────────────────────────────────────────
	async function pullModel(tag: string) {
		if (!tag || pullStates[tag]?.status === 'pulling') return;
		const tagsBefore = new Set(modelOptions().map((m) => m.model));
		pullStates[tag] = { status: 'pulling', progress: 'Starting…' };
		try {
			const res    = await fetch('/api/ollama/pull', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ model: tag })
			});
			const reader = res.body?.getReader();
			if (!reader) throw new Error('No response body');
			const dec = new TextDecoder();
			let buf = '';
			while (true) {
				const { done, value } = await reader.read();
				if (done) break;
				buf += dec.decode(value, { stream: true });
				const lines = buf.split('\n'); buf = lines.pop() ?? '';
				for (const line of lines) {
					if (!line.startsWith('data: ')) continue;
					const raw = line.slice(6).trim();
					if (raw === '[DONE]') break;
					try {
						const msg = JSON.parse(raw);
						if (msg.error) { pullStates[tag] = { status: 'error', progress: msg.error }; return; }
						if (msg.status) {
							const pct = msg.completed && msg.total
								? ` ${Math.round((msg.completed / msg.total) * 100)}%` : '';
							pullStates[tag] = { status: 'pulling', progress: msg.status + pct };
						}
					} catch { /* skip */ }
				}
			}
			await modelsStore.fetch(true);
			const resolvedTag = modelOptions().find((m) => !tagsBefore.has(m.model))?.model;
			pullStates[tag] = { status: 'done', progress: 'Installed', resolvedTag };
		} catch (e) {
			pullStates[tag] = { status: 'error', progress: e instanceof Error ? e.message : 'Failed' };
		}
	}

	async function useModel(tag: string) {
		try {
			const res  = await fetch('/api/set-model', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ model: tag })
			});
			const data = await res.json();
			if (isLlamacpp && data.status === 'starting') {
				await modelSwitchStore.startSwitch(tag);
			}
			await serverStore.fetch();
			await modelsStore.fetch(true);
		} catch (e) { console.error(e); }
	}

	function fmtSize(bytes: number): string {
		if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(1)} GB`;
		if (bytes >= 1e6) return `${(bytes / 1e6).toFixed(0)} MB`;
		return `${bytes} B`;
	}

	function fmtDownloads(n: number): string {
		if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
		if (n >= 1_000)     return `${(n / 1_000).toFixed(0)}k`;
		return String(n);
	}

	// ── derived lists ─────────────────────────────────────────────────────────────
	let catalogTags = $derived(new Set(catalog.flatMap((f) => f.variants.map((v) => v.tag))));
	let uncataloguedInstalled = $derived(installedModels.filter((m) => !catalogTags.has(m.model)));

	let filteredCatalog = $derived.by(() => {
		const q = searchQuery.trim().toLowerCase();
		if (!q) return catalog;
		return catalog.filter((f) =>
			f.name.toLowerCase().includes(q) ||
			f.description.toLowerCase().includes(q) ||
			f.variants.some((v) => v.tag.toLowerCase().includes(q))
		);
	});

	let filteredInstalled = $derived.by(() => {
		const q = searchQuery.trim().toLowerCase();
		if (!q) return installedModels;
		return installedModels.filter((m) => m.model.toLowerCase().includes(q));
	});

	// ── lifecycle ─────────────────────────────────────────────────────────────────
	$effect(() => {
		if (open) {
			modelsStore.fetch(true).catch(() => {});
			loadCatalog();
		}
	});
</script>

<Dialog.Root bind:open {onOpenChange}>
	<Dialog.Content class="flex max-h-[88vh] w-full max-w-xl flex-col gap-0 overflow-hidden p-0">

		<!-- Header -->
		<div class="flex items-center gap-3 border-b border-border px-4 py-3">
			<span class="text-sm font-semibold">Model Store</span>
			{#if activeTab !== 'huggingface'}
				<div class="relative flex-1">
					<Search class="pointer-events-none absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-muted-foreground" />
					<input
						type="text"
						placeholder={activeTab === 'installed' ? 'Filter installed models…' : 'Search models…'}
						bind:value={searchQuery}
						class="w-full rounded-md border border-border bg-muted/50 py-1.5 pl-8 pr-3 text-sm placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-ring"
					/>
				</div>
			{/if}
			<Dialog.Close class="ml-auto rounded-sm opacity-70 transition-opacity hover:opacity-100 focus:outline-none focus:ring-2 focus:ring-ring">
				<span class="sr-only">Close</span>✕
			</Dialog.Close>
		</div>

		<!-- Tabs -->
		<div class="flex border-b border-border text-sm">
			{#each ([['all', 'All Models'], ['installed', 'Installed'], ['huggingface', 'Hugging Face']] as const) as [id, label]}
				<button type="button"
					class={cn('px-4 py-2 font-medium transition',
						activeTab === id
							? 'border-b-2 border-foreground text-foreground'
							: 'text-muted-foreground hover:text-foreground')}
					onclick={() => { activeTab = id; ggufRepo = ''; }}>
					{label}
					{#if id === 'installed' && installedModels.length > 0}
						<span class="ml-1 rounded-full bg-muted px-1.5 py-0.5 text-xs text-muted-foreground">
							{installedModels.length}
						</span>
					{/if}
				</button>
			{/each}
		</div>

		<!-- Warning banner -->
		<div class="flex items-start gap-2 border-b border-amber-200 bg-amber-50 px-4 py-2 text-xs text-amber-800 dark:border-amber-900/40 dark:bg-amber-900/20 dark:text-amber-400">
			<AlertTriangle class="mt-0.5 h-3.5 w-3.5 shrink-0" />
			{#if isLlamacpp}
				<span>Models are downloaded as GGUF files to <code class="font-mono">/library/models</code>.</span>
			{:else}
				<span>Models are downloaded to your local machine via Ollama.</span>
			{/if}
		</div>

		<!-- ── Content ── -->
		<div class="min-h-0 flex-1 overflow-y-auto">

			<!-- ════ ALL MODELS ════ -->
			{#if activeTab === 'all'}
				{#if catalogLoading}
					<div class="flex items-center justify-center gap-2 py-16 text-muted-foreground">
						<Loader2 class="h-4 w-4 animate-spin" /><span class="text-sm">Loading…</span>
					</div>
				{:else}
					{#if uncataloguedInstalled.length > 0}
						<div class="px-4 pt-4 pb-2">
							<p class="mb-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">Installed (custom)</p>
							<div class="space-y-1">
								{#each uncataloguedInstalled as model (model.id)}
									{@const isActive = currentModel === model.model}
									<div class="flex items-center gap-2 rounded-md bg-muted/40 px-3 py-2">
										<span class="min-w-0 flex-1 truncate text-sm font-medium">{model.model}</span>
										{#if isActive}
											<span class="rounded-full bg-blue-500/10 px-2 py-0.5 text-xs font-medium text-blue-500">Active</span>
										{:else if switchingTo === model.model}
											<span class="flex items-center gap-1 text-xs text-muted-foreground">
												<Loader2 class="h-3 w-3 animate-spin" />Loading…
											</span>
										{:else}
											<button type="button"
												class="rounded bg-muted px-2 py-0.5 text-xs hover:bg-accent"
												onclick={() => useModel(model.model)}>Use</button>
										{/if}
									</div>
								{/each}
							</div>
						</div>
						<div class="mx-4 border-t border-border"></div>
					{/if}

					{#if filteredCatalog.length === 0}
						<div class="py-12 text-center text-sm text-muted-foreground">No models found.</div>
					{:else}
						<div class="divide-y divide-border">
							{#each filteredCatalog as family (family.name)}
								<div class="px-4 py-4">
									<div class="mb-0.5 text-sm font-semibold">{family.name}</div>
									<p class="mb-3 text-xs leading-relaxed text-muted-foreground">{family.description}</p>
									<div class="flex flex-wrap gap-2">
										{#each family.variants as variant (variant.tag)}
											{@const installed = installedTags.has(variant.tag)}
											{@const isActive  = currentModel === variant.tag}
											{@const ps        = pullStates[variant.tag]}
											{@const pulling   = ps?.status === 'pulling'}
											{@const pullErr   = ps?.status === 'error'}

											<button type="button"
												title={variant.description + (ps ? '\n' + ps.progress : '')}
												disabled={pulling || isLlamacpp}
												class={cn('inline-flex items-center gap-1 rounded px-2 py-0.5 text-xs font-medium transition',
													isActive  ? 'bg-foreground text-background' :
													installed ? 'bg-muted text-foreground hover:bg-accent' :
													pulling   ? 'cursor-wait bg-blue-500/10 text-blue-600' :
													pullErr   ? 'bg-red-500/10 text-red-500 hover:bg-red-500/20' :
													isLlamacpp ? 'cursor-default border border-dashed border-border bg-transparent text-muted-foreground/40' :
													             'border border-dashed border-border bg-transparent text-muted-foreground hover:border-solid hover:bg-muted hover:text-foreground'
												)}
												onclick={() => !isLlamacpp && (installed ? useModel(variant.tag) : pullModel(variant.tag))}>
												{#if pulling}
													<Loader2 class="h-2.5 w-2.5 animate-spin" />
												{:else if installed}
													<Check class="h-2.5 w-2.5 text-green-500" />
												{/if}
												{variant.params}
											</button>
										{/each}
									</div>
									{#if isLlamacpp}
										<p class="mt-2 text-[10px] text-muted-foreground/50">Use Hugging Face tab to download GGUF files.</p>
									{:else}
										<p class="mt-2 text-[10px] text-muted-foreground/60">
											<Check class="inline h-2.5 w-2.5 text-green-500" /> installed · dashed = available to install
										</p>
									{/if}
								</div>
							{/each}
						</div>
					{/if}
				{/if}

			<!-- ════ INSTALLED ════ -->
			{:else if activeTab === 'installed'}
				{#if filteredInstalled.length === 0}
					<div class="flex flex-col items-center gap-2 py-16 text-center">
						<p class="text-sm text-muted-foreground">
							{installedModels.length === 0 ? 'No LLM models installed yet.' : 'No models match your search.'}
						</p>
						{#if installedModels.length === 0}
							<button type="button" class="text-xs text-blue-500 hover:underline"
								onclick={() => (activeTab = 'huggingface')}>Browse models →</button>
						{/if}
					</div>
				{:else}
					<div class="divide-y divide-border">
						{#each filteredInstalled as model (model.id)}
							{@const isActive = currentModel === model.model}
							{@const det      = model.details}
							<div class="flex items-start gap-3 px-4 py-3">
								<div class="min-w-0 flex-1">
									<div class="flex flex-wrap items-center gap-2">
										<span class="text-sm font-medium">{model.model}</span>
										{#if isActive}
											<span class="rounded-full bg-blue-500/10 px-2 py-0.5 text-xs font-medium text-blue-500">Active</span>
										{/if}
									</div>
									<div class="mt-1 flex flex-wrap gap-x-3 gap-y-0.5 text-xs text-muted-foreground">
										{#if det?.parameter_size}<span>{det.parameter_size}</span>{/if}
										{#if det?.quantization_level}<span>{det.quantization_level}</span>{/if}
										{#if det?.family}<span class="capitalize">{det.family}</span>{/if}
										{#if det?.size}<span>{(Number(det.size) / 1e9).toFixed(1)} GB</span>{/if}
									</div>
								</div>
								{#if switchingTo === model.model}
									<span class="shrink-0 self-center flex items-center gap-1 text-xs text-muted-foreground">
										<Loader2 class="h-3 w-3 animate-spin" />Loading…
									</span>
								{:else if !isActive}
									<button type="button"
										class="shrink-0 self-center rounded-md bg-muted px-3 py-1.5 text-xs font-medium hover:bg-accent"
										onclick={() => useModel(model.model)}>Use</button>
								{/if}
							</div>
						{/each}
					</div>
				{/if}

			<!-- ════ HUGGING FACE ════ -->
			{:else}
				<div class="flex flex-col">
					<!-- HF search bar -->
					<div class="flex gap-2 border-b border-border px-4 py-3">
						{#if ggufRepo}
							<button type="button"
								class="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground"
								onclick={() => { ggufRepo = ''; ggufFiles = []; }}>
								<ChevronLeft class="h-3.5 w-3.5" />Back
							</button>
							<span class="flex-1 truncate self-center text-xs font-medium">{ggufRepo}</span>
						{:else}
							<div class="relative flex-1">
								<Search class="pointer-events-none absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-muted-foreground" />
								<input type="text"
									placeholder="Search GGUF models on Hugging Face…"
									bind:value={hfQuery}
									onkeydown={(e) => { if (e.key === 'Enter') searchHF(); }}
									class="w-full rounded-md border border-border bg-muted/50 py-1.5 pl-8 pr-3 text-sm placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-ring"
								/>
							</div>
							<button type="button"
								class="flex items-center gap-1.5 rounded-md bg-muted px-3 py-1.5 text-xs font-medium transition hover:bg-accent disabled:opacity-50"
								onclick={searchHF} disabled={hfLoading}>
								{#if hfLoading}<Loader2 class="h-3 w-3 animate-spin" />{:else}Search{/if}
							</button>
						{/if}
					</div>

					<!-- ── llamacpp: GGUF file picker ── -->
					{#if isLlamacpp && ggufRepo}
						{#if ggufLoading}
							<div class="flex items-center justify-center gap-2 py-16 text-muted-foreground">
								<Loader2 class="h-4 w-4 animate-spin" /><span class="text-sm">Loading files…</span>
							</div>
						{:else if ggufError}
							<div class="px-4 py-10 text-center text-sm text-red-500">{ggufError}</div>
						{:else if ggufFiles.length === 0}
							<div class="py-16 text-center text-sm text-muted-foreground">No GGUF files found in this repo.</div>
						{:else}
							<div class="divide-y divide-border">
								{#each ggufFiles as file (file.filename)}
									{@const dl        = dlStates[file.filename]}
									{@const installed = installedTags.has(file.filename)}
									{@const isActive  = currentModel === file.filename}
									<div class="flex items-center gap-3 px-4 py-3">
										<div class="min-w-0 flex-1">
											<span class="block truncate text-sm font-medium">{file.filename}</span>
											<div class="mt-0.5 flex items-center gap-3 text-xs text-muted-foreground">
												{#if file.size > 0}<span>{fmtSize(file.size)}</span>{/if}
												{#if dl?.status === 'downloading'}
													<span class="text-blue-500">{dl.msg} ({dl.pct}%)</span>
												{:else if dl?.status === 'error'}
													<span class="text-red-500">{dl.msg}</span>
												{/if}
											</div>
											{#if dl?.status === 'downloading'}
												<div class="mt-1.5 h-1 w-full overflow-hidden rounded-full bg-muted">
													<div class="h-full rounded-full bg-blue-500 transition-all" style="width:{dl.pct}%"></div>
												</div>
											{/if}
										</div>
										{#if isActive}
											<span class="shrink-0 text-xs text-muted-foreground">In use</span>
										{:else if switchingTo === file.filename}
											<span class="shrink-0 flex items-center gap-1 text-xs text-muted-foreground">
												<Loader2 class="h-3 w-3 animate-spin" />Loading…
											</span>
										{:else if installed}
											<button type="button"
												class="shrink-0 flex items-center gap-1 rounded-md bg-green-500/10 px-3 py-1.5 text-xs font-medium text-green-600 hover:bg-green-500/20"
												onclick={() => useModel(file.filename)}>
												<Check class="h-3 w-3" />Use
											</button>
										{:else if dl?.status === 'downloading'}
											<div class="shrink-0 flex items-center gap-1 text-xs text-blue-500">
												<Loader2 class="h-3 w-3 animate-spin" />
											</div>
										{:else if dl?.status === 'done'}
											<button type="button"
												class="shrink-0 flex items-center gap-1 rounded-md bg-green-500/10 px-3 py-1.5 text-xs font-medium text-green-600"
												onclick={() => useModel(file.filename)}>
												<Check class="h-3 w-3" />Use
											</button>
										{:else if dl?.status === 'error'}
											<button type="button"
												class="shrink-0 rounded-md bg-red-500/10 px-3 py-1.5 text-xs font-medium text-red-500 hover:bg-red-500/20"
												onclick={() => downloadGGUF(file.url, file.filename)}>Retry</button>
										{:else}
											<button type="button"
												class="shrink-0 flex items-center gap-1 rounded-md bg-blue-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-blue-700"
												onclick={() => downloadGGUF(file.url, file.filename)}>
												<Download class="h-3 w-3" />Download
											</button>
										{/if}
									</div>
								{/each}
							</div>
						{/if}

					<!-- ── HF search results ── -->
					{:else if hfLoading}
						<div class="flex items-center justify-center gap-2 py-16 text-muted-foreground">
							<Loader2 class="h-4 w-4 animate-spin" /><span class="text-sm">Searching Hugging Face…</span>
						</div>
					{:else if hfError}
						<div class="px-4 py-10 text-center text-sm text-red-500">{hfError}</div>
					{:else if hfResults.length === 0 && hfSearched}
						<div class="py-16 text-center text-sm text-muted-foreground">No GGUF models found.</div>
					{:else}
						<div class="divide-y divide-border">
							{#each hfResults as model (model.id)}
								{@const tag          = model.ollama_tag}
								{@const hfId         = tag.replace('hf.co/', '')}
								{@const ps           = pullStates[tag]}
								{@const resolvedTag  = ps?.resolvedTag ?? installedModels.find((m) => m.model.toLowerCase().includes(hfId.toLowerCase()))?.model}
								{@const installed    = !!(resolvedTag || installedTags.has(tag))}
								{@const isActive     = currentModel === tag || currentModel === resolvedTag}

								<div class="flex items-start gap-3 px-4 py-3">
									<div class="min-w-0 flex-1">
										<div class="flex flex-wrap items-center gap-2">
											<span class="text-sm font-medium">{model.id}</span>
											{#if isActive}
												<span class="rounded-full bg-blue-500/10 px-2 py-0.5 text-xs font-medium text-blue-500">Active</span>
											{:else if installed}
												<span class="rounded-full bg-green-500/10 px-2 py-0.5 text-xs font-medium text-green-600">Installed</span>
											{/if}
											{#if model.pipeline_tag}
												<span class="rounded bg-muted px-1.5 py-0.5 text-xs text-muted-foreground">{model.pipeline_tag}</span>
											{/if}
										</div>
										<div class="mt-0.5 flex items-center gap-3 text-xs text-muted-foreground">
											<span>↓ {fmtDownloads(model.downloads)}</span>
											{#if model.likes > 0}<span>♥ {fmtDownloads(model.likes)}</span>{/if}
											<a href="https://huggingface.co/{model.id}" target="_blank" rel="noopener noreferrer"
												class="inline-flex items-center gap-0.5 hover:text-foreground"
												onclick={(e) => e.stopPropagation()}>
												HF <ExternalLink class="h-2.5 w-2.5" />
											</a>
										</div>
										{#if !isLlamacpp}
											<p class="mt-0.5 font-mono text-[10px] text-muted-foreground/60">
												{resolvedTag || tag}
												{#if resolvedTag && resolvedTag !== tag}
													<span class="text-muted-foreground/40"> (stored as)</span>
												{/if}
											</p>
										{/if}
										{#if ps?.status === 'pulling' || ps?.status === 'error'}
											<p class={cn('mt-1 text-xs', ps.status === 'error' ? 'text-red-500' : 'text-blue-500')}>
												{ps.progress}
											</p>
										{/if}
									</div>

									{#if isLlamacpp}
										<!-- llamacpp: open repo file picker -->
										<button type="button"
											class="shrink-0 self-center flex items-center gap-1 rounded-md bg-muted px-3 py-1.5 text-xs font-medium hover:bg-accent"
											onclick={() => openRepo(model.id)}>
											Pick file
										</button>
									{:else if isActive}
										<span class="shrink-0 self-center text-xs text-muted-foreground">In use</span>
									{:else if installed}
										<button type="button"
											class="shrink-0 self-center rounded-md bg-muted px-3 py-1.5 text-xs font-medium hover:bg-accent"
											onclick={() => useModel(resolvedTag || tag)}>Use</button>
									{:else if ps?.status === 'pulling'}
										<button type="button" disabled
											class="shrink-0 self-center flex items-center gap-1 rounded-md bg-blue-500/10 px-3 py-1.5 text-xs font-medium text-blue-600">
											<Loader2 class="h-3 w-3 animate-spin" />Installing…
										</button>
									{:else if ps?.status === 'done'}
										<button type="button"
											class="shrink-0 self-center flex items-center gap-1 rounded-md bg-green-500/10 px-3 py-1.5 text-xs font-medium text-green-600"
											onclick={() => useModel(resolvedTag || tag)}>
											<Check class="h-3 w-3" />Use
										</button>
									{:else if ps?.status === 'error'}
										<button type="button"
											class="shrink-0 self-center rounded-md bg-red-500/10 px-3 py-1.5 text-xs font-medium text-red-500 hover:bg-red-500/20"
											onclick={() => pullModel(tag)}>Retry</button>
									{:else}
										<button type="button"
											class="shrink-0 self-center flex items-center gap-1 rounded-md bg-blue-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-blue-700"
											onclick={() => pullModel(tag)}>
											<Download class="h-3 w-3" />Install
										</button>
									{/if}
								</div>
							{/each}
						</div>
					{/if}
				</div>
			{/if}
		</div>

		<!-- Bottom bar — hidden in llamacpp mode (no custom pull) -->
		{#if !isLlamacpp}
			<div class="flex items-center gap-2 border-t border-border px-4 py-3">
				<input type="text"
					placeholder="Custom model (e.g. llama3:8b  or  hf.co/user/repo)"
					bind:value={customInput}
					onkeydown={(e) => { if (e.key === 'Enter') pullModel(customInput.trim()); }}
					class="min-w-0 flex-1 rounded-md border border-border bg-muted/50 px-3 py-1.5 text-xs placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-ring"
				/>
				<button type="button"
					class="flex items-center gap-1.5 rounded-md bg-blue-600 px-3 py-1.5 text-xs font-medium text-white transition hover:bg-blue-700 disabled:opacity-50"
					onclick={() => pullModel(customInput.trim())}
					disabled={!customInput.trim() || pullStates[customInput.trim()]?.status === 'pulling'}>
					{#if pullStates[customInput.trim()]?.status === 'pulling'}
						<Loader2 class="h-3 w-3 animate-spin" />{pullStates[customInput.trim()].progress}
					{:else}
						<Download class="h-3 w-3" />Download
					{/if}
				</button>
			</div>
		{/if}

	</Dialog.Content>
</Dialog.Root>
