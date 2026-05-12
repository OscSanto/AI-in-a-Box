<script lang="ts">
	import { page } from '$app/state';
	import { Plus, MessageSquare, Sparkles, Brain, RotateCcw, BookOpen, Palette } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import * as DropdownMenu from '$lib/components/ui/dropdown-menu';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import { FILE_TYPE_ICONS } from '$lib/constants/icons';
	import { TOOLTIP_DELAY_DURATION } from '$lib/constants/tooltip-config';
	import {
		ragControls,
		setBypassCache,
		toggleThinking,
		resetRagControls,
		DEFAULT_TONE,
		DEFAULT_FORMAT,
		TONE_LABELS,
		FORMAT_LABELS,
	} from '$lib/stores/ragControls.svelte';

	interface Props {
		class?: string;
		disabled?: boolean;
		hasAudioModality?: boolean;
		hasVisionModality?: boolean;
		supportsThinking?: boolean;
		onFileUpload?: () => void;
		onSystemPromptClick?: () => void;
	}

	let {
		class: className = '',
		disabled = false,
		hasAudioModality = false,
		hasVisionModality = false,
		supportsThinking = false,
		onFileUpload,
		onSystemPromptClick
	}: Props = $props();

	let currentThinking = $derived(ragControls().thinking);
	let activeZims = $derived(ragControls().activeZims);
	let zimLabel = $derived(activeZims.length === 0 ? 'All' : `${activeZims.length} selected`);

	let showKnowledgeBase = $state(false);
	let showResponseStyle = $state(false);

	let currentTone = $derived(ragControls().tone);
	let currentFormat = $derived(ragControls().format);
	let styleLabel = $derived(
		currentTone === DEFAULT_TONE && currentFormat === DEFAULT_FORMAT
			? 'Default'
			: `${TONE_LABELS[currentTone]} · ${FORMAT_LABELS[currentFormat]}`
	);

	let isNewChat = $derived(!page.params.id);

	let systemMessageTooltip = $derived(
		isNewChat
			? 'Add custom system message for a new conversation'
			: 'Inject custom system message at the beginning of the conversation'
	);

	let dropdownOpen = $state(false);

	const fileUploadTooltipText = 'Add files, system prompt or MCP Servers';
</script>

<div class="flex items-center gap-1 {className}">
	<DropdownMenu.Root bind:open={dropdownOpen}>
		<DropdownMenu.Trigger name="Attach files" {disabled}>
			<Tooltip.Root>
				<Tooltip.Trigger class="w-full">
					<Button
						class="file-upload-button h-8 w-8 rounded-full p-0"
						{disabled}
						variant="secondary"
						type="button"
					>
						<span class="sr-only">{fileUploadTooltipText}</span>

						<Plus class="h-4 w-4" />
					</Button>
				</Tooltip.Trigger>

				<Tooltip.Content>
					<p>{fileUploadTooltipText}</p>
				</Tooltip.Content>
			</Tooltip.Root>
		</DropdownMenu.Trigger>

		<DropdownMenu.Content align="start" class="w-64">
			{#if hasVisionModality}
				<DropdownMenu.Item
					class="images-button flex cursor-pointer items-center gap-2"
					onclick={() => onFileUpload?.()}
				>
					<FILE_TYPE_ICONS.image class="h-4 w-4" />

					<span>Images</span>
				</DropdownMenu.Item>
			{:else}
				<Tooltip.Root delayDuration={TOOLTIP_DELAY_DURATION}>
					<Tooltip.Trigger class="w-full">
						<DropdownMenu.Item
							class="images-button flex cursor-pointer items-center gap-2"
							disabled
						>
							<FILE_TYPE_ICONS.image class="h-4 w-4" />

							<span>Images</span>
						</DropdownMenu.Item>
					</Tooltip.Trigger>

					<Tooltip.Content side="right">
						<p>Images require vision models to be processed</p>
					</Tooltip.Content>
				</Tooltip.Root>
			{/if}

			{#if hasAudioModality}
				<DropdownMenu.Item
					class="audio-button flex cursor-pointer items-center gap-2"
					onclick={() => onFileUpload?.()}
				>
					<FILE_TYPE_ICONS.audio class="h-4 w-4" />

					<span>Audio Files</span>
				</DropdownMenu.Item>
			{:else}
				<Tooltip.Root delayDuration={TOOLTIP_DELAY_DURATION}>
					<Tooltip.Trigger class="w-full">
						<DropdownMenu.Item class="audio-button flex cursor-pointer items-center gap-2" disabled>
							<FILE_TYPE_ICONS.audio class="h-4 w-4" />

							<span>Audio Files</span>
						</DropdownMenu.Item>
					</Tooltip.Trigger>

					<Tooltip.Content side="right">
						<p>Audio files require audio models to be processed</p>
					</Tooltip.Content>
				</Tooltip.Root>
			{/if}

			<DropdownMenu.Item
				class="flex cursor-pointer items-center gap-2"
				onclick={() => onFileUpload?.()}
			>
				<FILE_TYPE_ICONS.text class="h-4 w-4" />

				<span>Text Files</span>
			</DropdownMenu.Item>

			{#if hasVisionModality}
				<DropdownMenu.Item
					class="flex cursor-pointer items-center gap-2"
					onclick={() => onFileUpload?.()}
				>
					<FILE_TYPE_ICONS.pdf class="h-4 w-4" />

					<span>PDF Files</span>
				</DropdownMenu.Item>
			{:else}
				<Tooltip.Root delayDuration={TOOLTIP_DELAY_DURATION}>
					<Tooltip.Trigger class="w-full">
						<DropdownMenu.Item
							class="flex cursor-pointer items-center gap-2"
							onclick={() => onFileUpload?.()}
						>
							<FILE_TYPE_ICONS.pdf class="h-4 w-4" />

							<span>PDF Files</span>
						</DropdownMenu.Item>
					</Tooltip.Trigger>

					<Tooltip.Content side="right">
						<p>PDFs will be converted to text. Image-based PDFs may not work properly.</p>
					</Tooltip.Content>
				</Tooltip.Root>
			{/if}

			<Tooltip.Root delayDuration={TOOLTIP_DELAY_DURATION}>
				<Tooltip.Trigger class="w-full">
					<DropdownMenu.Item
						class="flex cursor-pointer items-center gap-2"
						onclick={() => onSystemPromptClick?.()}
					>
						<MessageSquare class="h-4 w-4" />

						<span>System Message</span>
					</DropdownMenu.Item>
				</Tooltip.Trigger>

				<Tooltip.Content side="right">
					<p>{systemMessageTooltip}</p>
				</Tooltip.Content>
			</Tooltip.Root>

			<DropdownMenu.Separator />
			<DropdownMenu.Label class="text-[10px] uppercase tracking-wider text-muted-foreground/60">Commands</DropdownMenu.Label>

			<!-- Knowledge Base -->
			<DropdownMenu.Item
				class="flex cursor-pointer items-start gap-2 py-2"
				onclick={() => { dropdownOpen = false; showKnowledgeBase = true; }}
			>
				<BookOpen class="mt-0.5 h-3.5 w-3.5 flex-shrink-0" />
				<div class="min-w-0">
					<p class="text-xs font-medium">Knowledge Base</p>
					<p class="text-xs text-muted-foreground">ZIM libraries — {zimLabel}</p>
				</div>
			</DropdownMenu.Item>

			<!-- Response Style -->
			<DropdownMenu.Item
				class="flex cursor-pointer items-start gap-2 py-2"
				onclick={() => { dropdownOpen = false; showResponseStyle = true; }}
			>
				<Palette class="mt-0.5 h-3.5 w-3.5 flex-shrink-0" />
				<div class="min-w-0">
					<p class="text-xs font-medium">Response Style</p>
					<p class="text-xs text-muted-foreground">{styleLabel}</p>
				</div>
			</DropdownMenu.Item>

			<!-- /new — bypass cache -->
			<DropdownMenu.Item
				class="flex cursor-pointer items-start gap-2 py-2"
				onclick={() => setBypassCache(true)}
			>
				<Sparkles class="mt-0.5 h-3.5 w-3.5 flex-shrink-0" />
				<div class="min-w-0">
					<p class="font-mono text-xs">/new</p>
					<p class="text-xs text-muted-foreground">Skip cache — get a fresh answer</p>
				</div>
			</DropdownMenu.Item>

			<!-- /think — reasoning mode (disabled if model doesn't support it) -->
			{#if supportsThinking}
				<DropdownMenu.Item
					class="flex cursor-pointer items-start gap-2 py-2"
					onclick={() => toggleThinking()}
				>
					<Brain class="mt-0.5 h-3.5 w-3.5 flex-shrink-0 {currentThinking ? 'text-violet-500' : ''}" />
					<div class="min-w-0">
						<p class="font-mono text-xs">/think</p>
						<p class="text-xs text-muted-foreground">
							{currentThinking ? 'Reasoning on — click to disable' : 'Step-by-step reasoning before answering'}
						</p>
					</div>
				</DropdownMenu.Item>
			{:else}
				<Tooltip.Root delayDuration={TOOLTIP_DELAY_DURATION}>
					<Tooltip.Trigger class="w-full">
						<DropdownMenu.Item class="flex items-start gap-2 py-2 opacity-40" disabled>
							<Brain class="mt-0.5 h-3.5 w-3.5 flex-shrink-0" />
							<div class="min-w-0">
								<p class="font-mono text-xs">/think</p>
								<p class="text-xs text-muted-foreground">Step-by-step reasoning before answering</p>
							</div>
						</DropdownMenu.Item>
					</Tooltip.Trigger>
					<Tooltip.Content side="right">
						<p>Only available with reasoning models (e.g. DeepSeek-R1, QwQ)</p>
					</Tooltip.Content>
				</Tooltip.Root>
			{/if}

			<!-- /reset — reset all controls -->
			<DropdownMenu.Item
				class="flex cursor-pointer items-start gap-2 py-2"
				onclick={() => resetRagControls()}
			>
				<RotateCcw class="mt-0.5 h-3.5 w-3.5 flex-shrink-0" />
				<div class="min-w-0">
					<p class="font-mono text-xs">/reset</p>
					<p class="text-xs text-muted-foreground">Reset mode, ZIM filter, and cache settings</p>
				</div>
			</DropdownMenu.Item>
		</DropdownMenu.Content>
	</DropdownMenu.Root>
</div>

{#if showKnowledgeBase}
	{#await import('$lib/components/app/dialogs/DialogKnowledgeBase.svelte') then { default: DialogKnowledgeBase }}
		<DialogKnowledgeBase bind:open={showKnowledgeBase} />
	{/await}
{/if}

{#if showResponseStyle}
	{#await import('$lib/components/app/dialogs/DialogResponseStyle.svelte') then { default: DialogResponseStyle }}
		<DialogResponseStyle bind:open={showResponseStyle} />
	{/await}
{/if}
