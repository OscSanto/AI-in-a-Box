<script lang="ts">
	import { onMount } from 'svelte';
	import { BookOpen, Loader2 } from '@lucide/svelte';
	import * as Dialog from '$lib/components/ui/dialog';
	import { ragControls, setActiveZims, toggleZim, setRagMode } from '$lib/stores/ragControls.svelte';

	interface Props {
		open?: boolean;
		onOpenChange?: (open: boolean) => void;
	}

	let { open = $bindable(), onOpenChange }: Props = $props();

	interface ZimEntry {
		name: string;
		enabled: boolean;
		count: number;
	}

	let zims = $state<ZimEntry[]>([]);
	let loading = $state(false);
	let error = $state('');

	let activeZims = $derived(ragControls().activeZims);
	let currentMode = $derived(ragControls().mode);
	let zimEnabled = $derived(currentMode === 'balanced');
	let allZimNames = $derived(zims.map((z) => z.name));

	function isZimActive(name: string): boolean {
		// empty activeZims = all active
		return activeZims.length === 0 || activeZims.includes(name);
	}

	async function loadZims() {
		loading = true;
		error = '';
		try {
			const res = await fetch('/api/libraries');
			if (!res.ok) throw new Error(`Server error ${res.status}`);
			const data = await res.json();
			zims = (data.libraries ?? []) as ZimEntry[];
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to load libraries';
		} finally {
			loading = false;
		}
	}

	$effect(() => {
		if (open) loadZims();
	});

	function handleToggle(name: string) {
		toggleZim(name, allZimNames);
	}

	function handleEnableAll() {
		setActiveZims([]);
	}
</script>

<Dialog.Root bind:open {onOpenChange}>
	<Dialog.Content class="flex max-h-[88vh] w-full max-w-lg flex-col gap-0 overflow-hidden p-0">

		<!-- Header -->
		<div class="flex items-center gap-3 border-b border-border px-4 py-3">
			<BookOpen class="h-4 w-4 text-muted-foreground" />
			<span class="text-sm font-semibold">Knowledge Base</span>
			<Dialog.Close class="ml-auto rounded-sm opacity-70 transition-opacity hover:opacity-100 focus:outline-none focus:ring-2 focus:ring-ring">
				<span class="sr-only">Close</span>✕
			</Dialog.Close>
		</div>

		<!-- ZIM on/off toggle -->
		<div class="flex items-center justify-between border-b border-border px-4 py-3">
			<div>
				<p class="text-sm font-medium">Use ZIM libraries</p>
				<p class="text-xs text-muted-foreground">Search offline content before answering</p>
			</div>
			<button
				type="button"
				onclick={() => setRagMode(zimEnabled ? 'chat' : 'balanced')}
				class="relative inline-flex h-6 w-11 shrink-0 items-center rounded-full transition {zimEnabled ? 'bg-blue-600' : 'bg-muted-foreground/30'}"
			>
				<span class="inline-block h-4 w-4 transform rounded-full bg-white shadow transition {zimEnabled ? 'translate-x-6' : 'translate-x-1'}"></span>
			</button>
		</div>

		<!-- Body -->
		<div class="min-h-0 flex-1 overflow-y-auto">

			<!-- ZIM indexes -->
			<div class="px-4 pt-4 pb-2 {zimEnabled ? '' : 'pointer-events-none opacity-40'}">
				<div class="mb-2 flex items-center justify-between">
					<p class="text-xs font-semibold uppercase tracking-wide text-muted-foreground">ZIM Libraries</p>
					{#if activeZims.length > 0}
						<button
							type="button"
							class="text-xs text-blue-500 hover:underline"
							onclick={handleEnableAll}
						>
							Enable all
						</button>
					{/if}
				</div>

				{#if loading}
					<div class="flex items-center justify-center gap-2 py-10 text-muted-foreground">
						<Loader2 class="h-4 w-4 animate-spin" /><span class="text-sm">Loading…</span>
					</div>
				{:else if error}
					<p class="py-6 text-center text-sm text-red-500">{error}</p>
				{:else if zims.length === 0}
					<p class="py-6 text-center text-sm text-muted-foreground">No ZIM libraries found.</p>
				{:else}
					<div class="space-y-1">
						{#each zims as zim (zim.name)}
							{@const active = isZimActive(zim.name)}
							<button
								type="button"
								class="flex w-full items-center gap-3 rounded-md px-3 py-2.5 text-left transition hover:bg-muted/60 {active ? '' : 'opacity-50'}"
								onclick={() => handleToggle(zim.name)}
							>
								<!-- toggle indicator -->
								<span class="relative inline-flex h-5 w-9 shrink-0 items-center rounded-full transition {active ? 'bg-blue-600' : 'bg-muted-foreground/30'}">
									<span class="inline-block h-3.5 w-3.5 transform rounded-full bg-white shadow transition {active ? 'translate-x-4' : 'translate-x-0.5'}"></span>
								</span>
								<div class="min-w-0 flex-1">
									<p class="truncate text-sm font-medium">{zim.name}</p>
									<p class="text-xs text-muted-foreground">{zim.count} chunk{zim.count !== 1 ? 's' : ''} per query</p>
								</div>
							</button>
						{/each}
					</div>
				{/if}
			</div>
		</div>

		<!-- Footer note -->
		<div class="border-t border-border px-4 py-3 text-xs text-muted-foreground">
			Toggled libraries apply to all queries in this session. Disabled libraries are skipped during retrieval.
		</div>

	</Dialog.Content>
</Dialog.Root>
