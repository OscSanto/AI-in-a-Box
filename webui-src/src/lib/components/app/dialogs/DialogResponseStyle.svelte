<script lang="ts">
	import { Palette } from '@lucide/svelte';
	import * as Dialog from '$lib/components/ui/dialog';
	import {
		ragControls,
		setTone,
		setFormat,
		DEFAULT_TONE,
		DEFAULT_FORMAT,
		TONE_LABELS,
		FORMAT_LABELS,
		TONE_DESCRIPTIONS,
		FORMAT_DESCRIPTIONS,
		type RagTone,
		type RagFormat,
	} from '$lib/stores/ragControls.svelte';

	interface Props {
		open?: boolean;
		onOpenChange?: (open: boolean) => void;
	}

	let { open = $bindable(), onOpenChange }: Props = $props();

	const TONES: RagTone[] = ['neutral', 'friendly', 'socratic'];
	const FORMATS: RagFormat[] = ['prose', 'structured', 'direct'];

	let currentTone = $derived(ragControls().tone);
	let currentFormat = $derived(ragControls().format);
</script>

<Dialog.Root bind:open {onOpenChange}>
	<Dialog.Content class="flex max-h-[88vh] w-full max-w-lg flex-col gap-0 overflow-hidden p-0">

		<!-- Header -->
		<div class="flex items-center gap-3 border-b border-border px-4 py-3">
			<Palette class="h-4 w-4 text-muted-foreground" />
			<span class="text-sm font-semibold">Response Style</span>
			<Dialog.Close class="ml-auto rounded-sm opacity-70 transition-opacity hover:opacity-100 focus:outline-none focus:ring-2 focus:ring-ring">
				<span class="sr-only">Close</span>✕
			</Dialog.Close>
		</div>

		<!-- Body -->
		<div class="min-h-0 flex-1 overflow-y-auto px-4 py-4 space-y-5">

			<!-- Tone -->
			<div>
				<p class="mb-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">Tone</p>
				<div class="grid grid-cols-3 gap-2">
					{#each TONES as tone}
						<button
							type="button"
							class="rounded-lg border p-3 text-left transition {currentTone === tone
								? 'border-blue-500 bg-blue-500/10'
								: 'border-border hover:bg-muted/60'}"
							onclick={() => setTone(tone)}
						>
							<p class="text-xs font-semibold">{TONE_LABELS[tone]}</p>
							<p class="mt-1 text-xs text-muted-foreground leading-snug">{TONE_DESCRIPTIONS[tone]}</p>
						</button>
					{/each}
				</div>
			</div>

			<!-- Format -->
			<div>
				<p class="mb-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">Format</p>
				<div class="grid grid-cols-3 gap-2">
					{#each FORMATS as fmt}
						<button
							type="button"
							class="rounded-lg border p-3 text-left transition {currentFormat === fmt
								? 'border-blue-500 bg-blue-500/10'
								: 'border-border hover:bg-muted/60'}"
							onclick={() => setFormat(fmt)}
						>
							<p class="text-xs font-semibold">{FORMAT_LABELS[fmt]}</p>
							<p class="mt-1 text-xs text-muted-foreground leading-snug">{FORMAT_DESCRIPTIONS[fmt]}</p>
						</button>
					{/each}
				</div>
			</div>
		</div>

		<!-- Footer -->
		<div class="border-t border-border px-4 py-3 text-xs text-muted-foreground">
			Style applies to all queries in this session and is saved for future sessions.
			{#if currentTone !== DEFAULT_TONE || currentFormat !== DEFAULT_FORMAT}
				<button
					type="button"
					class="ml-1 text-blue-500 hover:underline"
					onclick={() => { setTone(DEFAULT_TONE); setFormat(DEFAULT_FORMAT); }}
				>Reset to default</button>
			{/if}
		</div>

	</Dialog.Content>
</Dialog.Root>
