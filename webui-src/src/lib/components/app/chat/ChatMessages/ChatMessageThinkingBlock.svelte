<script lang="ts">
	import { Loader, ChevronUp, ChevronDown } from '@lucide/svelte';
	import * as Collapsible from '$lib/components/ui/collapsible/index.js';
	import { config } from '$lib/stores/settings.svelte';

	interface Props {
		class?: string;
		hasRegularContent?: boolean;
		isStreaming?: boolean;
		reasoningContent: string | null;
	}

	let {
		class: className = '',
		hasRegularContent = false,
		isStreaming = false,
		reasoningContent
	}: Props = $props();

	const currentConfig = config();

	let isExpanded = $state(currentConfig.showThoughtInProgress);

	$effect(() => {
		if (hasRegularContent && reasoningContent && currentConfig.showThoughtInProgress) {
			isExpanded = false;
		}
	});
</script>

<Collapsible.Root bind:open={isExpanded} class="mb-4 {className}">
	<Collapsible.Trigger class="flex w-full cursor-pointer items-center gap-2 text-muted-foreground hover:text-foreground/70">
		{#if isStreaming}
			<Loader class="h-3.5 w-3.5 animate-spin" />
		{:else}
			<!-- Static dot when done -->
			<span class="h-3.5 w-3.5 flex items-center justify-center">
				<span class="h-1.5 w-1.5 rounded-full bg-muted-foreground/50"></span>
			</span>
		{/if}

		<span class="text-sm text-muted-foreground/80">
			{isStreaming ? 'thinking' : 'thinking'}
		</span>

		<span class="ml-auto">
			{#if isExpanded}
				<ChevronUp class="h-3.5 w-3.5 text-muted-foreground/60" />
			{:else}
				<ChevronDown class="h-3.5 w-3.5 text-muted-foreground/60" />
			{/if}
		</span>
	</Collapsible.Trigger>

	<Collapsible.Content>
		<div class="mt-2 pl-5 text-sm leading-relaxed text-muted-foreground/70 break-words">
			{reasoningContent ?? ''}
		</div>
	</Collapsible.Content>
</Collapsible.Root>
