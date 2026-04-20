<script lang="ts">
	import { onMount } from 'svelte';
	import { Plus, Trash2, CheckCircle, XCircle, Loader, RefreshCw, Wifi, ChevronUp, ChevronDown, Zap } from '@lucide/svelte';
	import { inferenceBackendsStore, type InferenceBackend } from '$lib/stores/inferenceBackends.svelte';

	// Local editable copy of user backends only (builtins are read-only)
	let editableBackends = $state<InferenceBackend[]>([]);
	let testResults = $state<Record<string, { healthy: boolean; models: string[]; testing: boolean }>>({});
	let saveStatus = $state<'idle' | 'saving' | 'saved'>('idle');

	// Flash Attention state
	let flashAttention = $state(false);
	let flashSaveStatus = $state<'idle' | 'saving' | 'saved'>('idle');

	onMount(async () => {
		await inferenceBackendsStore.load();
		editableBackends = inferenceBackendsStore.userBackends.map((b) => ({ ...b }));

		try {
			const res = await fetch('/api/settings/performance');
			if (res.ok) {
				const data = await res.json();
				flashAttention = data.flash_attention ?? false;
			}
		} catch { /* leave at default */ }
	});

	async function saveFlashAttention() {
		flashSaveStatus = 'saving';
		try {
			const res = await fetch('/api/settings/performance', {
				method: 'PATCH',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ flash_attention: flashAttention }),
			});
			if (res.ok) {
				const data = await res.json();
				flashAttention = data.flash_attention;
				flashSaveStatus = 'saved';
				setTimeout(() => (flashSaveStatus = 'idle'), 2500);
			}
		} catch {
			flashSaveStatus = 'idle';
		}
	}

	function addBackend() {
		editableBackends = [
			...editableBackends,
			{
				id: `user-new-${Date.now()}`,
				name: '',
				url: '',
				priority: 10 + editableBackends.length,
				builtin: false,
				enabled: true
			}
		];
	}

	function removeBackend(id: string) {
		editableBackends = editableBackends.filter((b) => b.id !== id);
		const { [id]: _, ...rest } = testResults;
		testResults = rest;
	}

	function moveBackend(index: number, direction: -1 | 1) {
		const swapIndex = index + direction;
		if (swapIndex < 0 || swapIndex >= editableBackends.length) return;
		const arr = [...editableBackends];
		[arr[index], arr[swapIndex]] = [arr[swapIndex], arr[index]];
		// Reassign priorities to match display order
		editableBackends = arr.map((b, i) => ({ ...b, priority: 10 + i }));
	}

	async function testBackend(backend: InferenceBackend) {
		if (!backend.url.trim()) return;
		testResults = { ...testResults, [backend.id]: { healthy: false, models: [], testing: true } };
		const result = await inferenceBackendsStore.testUrl(backend.url.trim());
		testResults = { ...testResults, [backend.id]: { ...result, testing: false } };
	}

	async function saveAll() {
		saveStatus = 'saving';
		await inferenceBackendsStore.save(editableBackends.filter((b) => b.url.trim()));
		saveStatus = 'saved';
		setTimeout(() => (saveStatus = 'idle'), 2000);
	}

	// Builtin backends for display (read-only)
	const builtinBackends = $derived(
		inferenceBackendsStore.backends.filter((b) => b.builtin)
	);
</script>

<div class="space-y-6">
	<!-- Flash Attention -->
	<div class="rounded-lg border border-border/40 p-4">
		<div class="mb-3 flex items-center gap-2">
			<Zap class="h-4 w-4 text-muted-foreground" />
			<h4 class="text-sm font-medium">Ollama Performance</h4>
		</div>
		<div class="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
			<div class="min-w-0 flex-1">
				<p class="text-sm font-medium">Flash Attention</p>
				<p class="mt-0.5 text-xs text-muted-foreground">
					Reduces VRAM usage during inference using the FlashAttention kernel.
					Primarily beneficial for GPU backends — minimal effect on CPU-only devices like Raspberry Pi.
					Changes take effect after restarting Ollama.
				</p>
				{#if flashSaveStatus === 'saved'}
					<p class="mt-1.5 flex items-center gap-1 text-xs text-amber-500 dark:text-amber-400">
						<CheckCircle class="h-3 w-3" />
						Saved — restart Ollama to apply:
						<code class="rounded bg-muted px-1 font-mono">sudo systemctl restart ollama</code>
					</p>
				{/if}
			</div>
			<div class="flex shrink-0 items-center gap-2 self-start">
				<input
					type="checkbox"
					id="flash-attention-toggle"
					bind:checked={flashAttention}
					class="h-4 w-4 accent-primary"
				/>
				<button
					class="flex items-center gap-1.5 rounded-md bg-primary px-3 py-1 text-xs font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-60"
					onclick={saveFlashAttention}
					disabled={flashSaveStatus === 'saving'}
				>
					{#if flashSaveStatus === 'saving'}
						<Loader class="h-3 w-3 animate-spin" />
					{:else}
						Save
					{/if}
				</button>
			</div>
		</div>
	</div>

	<!-- Built-in backends (read-only status display) -->
	{#if builtinBackends.length > 0}
		<div>
			<h4 class="mb-2 text-sm font-medium text-muted-foreground">Built-in Backends</h4>
			<div class="space-y-2">
				{#each builtinBackends as backend (backend.id)}
					<div class="flex items-center gap-3 rounded-lg border border-border/40 bg-muted/30 px-3 py-2">
						<Wifi class="h-4 w-4 shrink-0 text-muted-foreground" />
						<div class="min-w-0 flex-1">
							<div class="flex items-center gap-2">
								<span class="text-sm font-medium">{backend.name}</span>
								<span class="rounded bg-muted px-1.5 py-0.5 text-[10px] text-muted-foreground">
									{backend.id === 'pi-local' ? 'fallback' : 'auto-discovered'}
								</span>
							</div>
							<p class="truncate text-xs text-muted-foreground">{backend.url || 'Not connected'}</p>
						</div>
						<div class="shrink-0">
							{#if !backend.url}
								<span class="text-xs text-muted-foreground">offline</span>
							{:else if backend.healthy === true}
								<CheckCircle class="h-4 w-4 text-green-500" />
							{:else if backend.healthy === false}
								<XCircle class="h-4 w-4 text-muted-foreground" />
							{/if}
						</div>
					</div>
				{/each}
			</div>
		</div>
	{/if}

	<!-- User-configured backends -->
	<div>
		<div class="mb-2 flex items-center justify-between">
			<h4 class="text-sm font-medium text-muted-foreground">Remote Backends</h4>
			<button
				class="flex items-center gap-1 rounded-md px-2 py-1 text-xs text-muted-foreground hover:bg-accent hover:text-accent-foreground"
				onclick={() => inferenceBackendsStore.load()}
			>
				<RefreshCw class="h-3 w-3" />
				Refresh
			</button>
		</div>

		{#if editableBackends.length === 0}
			<p class="rounded-lg border border-dashed border-border/40 px-4 py-6 text-center text-sm text-muted-foreground">
				No remote backends configured.<br />
				Add a GPU machine or hosted Ollama instance below.
			</p>
		{:else}
			<div class="space-y-3">
				{#each editableBackends as backend (backend.id)}
					{@const test = testResults[backend.id]}
					<div class="rounded-lg border border-border/40 p-3">
						<div class="flex items-start gap-2">
							<!-- Enable toggle -->
							<input
								type="checkbox"
								bind:checked={backend.enabled}
								class="mt-1 h-4 w-4 shrink-0 accent-primary"
								title="Enable this backend"
							/>

							<div class="min-w-0 flex-1 space-y-2">
								<!-- Name + URL row -->
								<div class="flex flex-col gap-2 sm:flex-row">
									<input
										type="text"
										bind:value={backend.name}
										placeholder="Name (e.g. Home GPU)"
										class="w-full shrink-0 rounded-md border border-border/60 bg-background px-2 py-1 text-sm outline-none focus:border-primary sm:w-28"
									/>
									<input
										type="text"
										bind:value={backend.url}
										placeholder="http://192.168.x.x:11434"
										class="min-w-0 flex-1 rounded-md border border-border/60 bg-background px-2 py-1 font-mono text-xs outline-none focus:border-primary"
									/>
								</div>

								<!-- Test result -->
								{#if test}
									{#if test.testing}
										<div class="flex items-center gap-1.5 text-xs text-muted-foreground">
											<Loader class="h-3 w-3 animate-spin" />
											Testing…
										</div>
									{:else if test.healthy}
										<div class="flex items-center gap-1.5 text-xs text-green-600 dark:text-green-400">
											<CheckCircle class="h-3 w-3" />
											Reachable
											{#if test.models.length > 0}
												— {test.models.slice(0, 3).join(', ')}{test.models.length > 3 ? ` +${test.models.length - 3} more` : ''}
											{/if}
										</div>
									{:else}
										<div class="flex items-center gap-1.5 text-xs text-red-500">
											<XCircle class="h-3 w-3" />
											Unreachable — check URL and Ollama is running
										</div>
									{/if}
								{/if}
							</div>

							<!-- Actions -->
							<div class="flex shrink-0 gap-1">
								<!-- Priority up/down -->
								<div class="flex flex-col">
									<button
										class="rounded px-1 py-0.5 text-muted-foreground hover:bg-accent hover:text-accent-foreground disabled:opacity-20"
										onclick={() => moveBackend(editableBackends.indexOf(backend), -1)}
										disabled={editableBackends.indexOf(backend) === 0}
										title="Higher priority (tried first)"
									>
										<ChevronUp class="h-3 w-3" />
									</button>
									<button
										class="rounded px-1 py-0.5 text-muted-foreground hover:bg-accent hover:text-accent-foreground disabled:opacity-20"
										onclick={() => moveBackend(editableBackends.indexOf(backend), 1)}
										disabled={editableBackends.indexOf(backend) === editableBackends.length - 1}
										title="Lower priority (tried later)"
									>
										<ChevronDown class="h-3 w-3" />
									</button>
								</div>
								<button
									class="rounded p-1.5 text-muted-foreground hover:bg-accent hover:text-accent-foreground disabled:opacity-40"
									onclick={() => testBackend(backend)}
									disabled={!backend.url.trim() || test?.testing}
									title="Test connection"
								>
									{#if test?.testing}
										<Loader class="h-3.5 w-3.5 animate-spin" />
									{:else}
										<Wifi class="h-3.5 w-3.5" />
									{/if}
								</button>
								<button
									class="rounded p-1.5 text-muted-foreground hover:bg-destructive/10 hover:text-destructive"
									onclick={() => removeBackend(backend.id)}
									title="Remove"
								>
									<Trash2 class="h-3.5 w-3.5" />
								</button>
							</div>
						</div>
					</div>
				{/each}
			</div>
		{/if}

		<button
			class="mt-3 flex w-full items-center justify-center gap-2 rounded-lg border border-dashed border-border/60 py-2 text-sm text-muted-foreground hover:border-primary/40 hover:text-foreground"
			onclick={addBackend}
		>
			<Plus class="h-4 w-4" />
			Add backend
		</button>
	</div>

	<!-- Priority note -->
	<p class="text-xs text-muted-foreground">
		Backends are tried top-to-bottom. The first one that responds is used.
		Pi Local is always the last fallback. Use ↑↓ to reorder.
		The active backend appears in the header after each response.
	</p>

	<!-- Save button -->
	<div class="flex justify-end">
		<button
			class="flex items-center gap-2 rounded-md bg-primary px-4 py-1.5 text-sm font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-60"
			onclick={saveAll}
			disabled={saveStatus === 'saving'}
		>
			{#if saveStatus === 'saving'}
				<Loader class="h-3.5 w-3.5 animate-spin" />
				Saving…
			{:else if saveStatus === 'saved'}
				<CheckCircle class="h-3.5 w-3.5" />
				Saved
			{:else}
				Save backends
			{/if}
		</button>
	</div>
</div>
