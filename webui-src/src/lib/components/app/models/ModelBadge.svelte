<script lang="ts">
	import { Package, Loader2 } from '@lucide/svelte';
	import { BadgeInfo, ActionIconCopyToClipboard } from '$lib/components/app';
	import { modelsStore } from '$lib/stores/models.svelte';
	import { serverStore } from '$lib/stores/server.svelte';
	import { modelSwitchStore } from '$lib/stores/modelSwitch.svelte';
	import * as Tooltip from '$lib/components/ui/tooltip';

	interface Props {
		class?: string;
		model?: string;
		onclick?: () => void;
		showCopyIcon?: boolean;
		showTooltip?: boolean;
	}

	let {
		class: className = '',
		model: modelProp,
		onclick,
		showCopyIcon = false,
		showTooltip = false
	}: Props = $props();

	let model       = $derived(modelProp || modelsStore.singleModelName);
	let isModelMode = $derived(serverStore.isModelMode);
	let isSwitching = $derived(modelSwitchStore.isSwitching);
	let switchingTo = $derived(modelSwitchStore.switchingTo);
</script>

{#snippet badgeContent()}
	<BadgeInfo class={className} {onclick}>
		{#snippet icon()}
			{#if isSwitching}
				<Loader2 class="h-3 w-3 animate-spin" />
			{:else}
				<Package class="h-3 w-3" />
			{/if}
		{/snippet}

		{#if isSwitching}
			<span class="text-muted-foreground">Loading {switchingTo}…</span>
		{:else}
			{model}
		{/if}

		{#if showCopyIcon && !isSwitching}
			<ActionIconCopyToClipboard text={model || ''} ariaLabel="Copy model name" />
		{/if}
	</BadgeInfo>
{/snippet}

{#if model && isModelMode}
	{#if showTooltip}
		<Tooltip.Root>
			<Tooltip.Trigger>
				{@render badgeContent()}
			</Tooltip.Trigger>

			<Tooltip.Content>
				{onclick ? 'Click for model details' : model}
			</Tooltip.Content>
		</Tooltip.Root>
	{:else}
		{@render badgeContent()}
	{/if}
{/if}
