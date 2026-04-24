<script lang="ts">
	import { Square } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import {
		ChatFormActionAttachmentsDropdown,
		ChatFormActionRecord,
		ChatFormActionSubmit,
	} from '$lib/components/app';
	import { FileTypeCategory } from '$lib/enums';
	import { getFileTypeCategory } from '$lib/utils';
	import { config } from '$lib/stores/settings.svelte';
	import { modelsStore, modelOptions, selectedModelId } from '$lib/stores/models.svelte';
	import { isRouterMode } from '$lib/stores/server.svelte';
	import { chatStore } from '$lib/stores/chat.svelte';
	import { activeMessages } from '$lib/stores/conversations.svelte';
	import {
		ragControls,
		setRagMode,
		type RagMode,
		modelSupportsThinking
	} from '$lib/stores/ragControls.svelte';

	let currentMode = $derived(ragControls().mode);

	const MODES: { value: RagMode; label: string }[] = [
		{ value: 'fast',     label: 'Fast'     },
		{ value: 'balanced', label: 'Balanced' },
		{ value: 'complex',  label: 'Complex'  },
		{ value: 'chat',     label: 'Chat'     },
	];

	interface Props {
		canSend?: boolean;
		class?: string;
		disabled?: boolean;
		isLoading?: boolean;
		isRecording?: boolean;
		hasText?: boolean;
		uploadedFiles?: ChatUploadedFile[];
		onFileUpload?: () => void;
		onMicClick?: () => void;
		onStop?: () => void;
		onSystemPromptClick?: () => void;
	}

	let {
		canSend = false,
		class: className = '',
		disabled = false,
		isLoading = false,
		isRecording = false,
		hasText = false,
		uploadedFiles = [],
		onFileUpload,
		onMicClick,
		onStop,
		onSystemPromptClick
	}: Props = $props();

	let currentConfig = $derived(config());
	let isRouter = $derived(isRouterMode());

	let conversationModel = $derived(
		chatStore.getConversationModel(activeMessages() as DatabaseMessage[])
	);

	let previousConversationModel: string | null = null;

	$effect(() => {
		if (conversationModel && conversationModel !== previousConversationModel) {
			previousConversationModel = conversationModel;
			modelsStore.selectModelByName(conversationModel);
		}
	});

	let activeModelId = $derived.by(() => {
		const options = modelOptions();

		if (!isRouter) {
			return options.length > 0 ? options[0].model : null;
		}

		const selectedId = selectedModelId();
		if (selectedId) {
			const model = options.find((m) => m.id === selectedId);
			if (model) return model.model;
		}

		if (conversationModel) {
			const model = options.find((m) => m.model === conversationModel);
			if (model) return model.model;
		}

		return null;
	});

	let modelPropsVersion = $state(0); // Used to trigger reactivity after fetch

	$effect(() => {
		if (activeModelId) {
			const cached = modelsStore.getModelProps(activeModelId);

			if (!cached) {
				modelsStore.fetchModelProps(activeModelId).then(() => {
					modelPropsVersion++;
				});
			}
		}
	});

	let hasAudioModality = $derived.by(() => {
		if (activeModelId) {
			void modelPropsVersion;

			return modelsStore.modelSupportsAudio(activeModelId);
		}

		return false;
	});

	let hasVisionModality = $derived.by(() => {
		if (activeModelId) {
			void modelPropsVersion;

			return modelsStore.modelSupportsVision(activeModelId);
		}

		return false;
	});

	let hasAudioAttachments = $derived(
		uploadedFiles.some((file) => getFileTypeCategory(file.type) === FileTypeCategory.AUDIO)
	);
	let shouldShowRecordButton = $derived(
		hasAudioModality && !hasText && !hasAudioAttachments && currentConfig.autoMicOnEmpty
	);

	let hasModelSelected = $derived(!isRouter || !!conversationModel || !!selectedModelId());

	let isSelectedModelInCache = $derived.by(() => {
		if (!isRouter) return true;

		if (conversationModel) {
			return modelOptions().some((option) => option.model === conversationModel);
		}

		const currentModelId = selectedModelId();
		if (!currentModelId) return false;

		return modelOptions().some((option) => option.id === currentModelId);
	});

	let activeModelName = $derived(isRouter ? activeModelId : activeModelId);
	let supportsThinking = $derived(modelSupportsThinking(activeModelName));

	let submitTooltip = $derived.by(() => {
		if (!hasModelSelected) {
			return 'Please select a model first';
		}

		if (!isSelectedModelInCache) {
			return 'Selected model is not available, please select another';
		}

		return '';
	});

	export function openModelSelector() {
		// No-op: model selector removed from chat form
	}
</script>

<div class="flex w-full items-center gap-3 {className}" style="container-type: inline-size">
	<div class="mr-auto flex items-center gap-2">
		<ChatFormActionAttachmentsDropdown
			{disabled}
			{hasAudioModality}
			{hasVisionModality}
			{supportsThinking}
			{onFileUpload}
			{onSystemPromptClick}
		/>

		<!-- Pipeline mode buttons -->
		<div class="flex items-center gap-0.5 rounded-full border border-border bg-muted/40 p-0.5">
			{#each MODES as m}
				<button
					type="button"
					disabled={disabled}
					onclick={() => setRagMode(m.value)}
					class="rounded-full px-2 py-0.5 text-[10px] font-medium transition-colors sm:px-2.5 sm:text-xs
						{currentMode === m.value
							? 'bg-primary text-primary-foreground shadow-sm'
							: 'text-muted-foreground hover:text-foreground'}"
				>
					{m.label}
				</button>
			{/each}
		</div>
	</div>

	{#if isLoading}
		<Button
			type="button"
			variant="secondary"
			onclick={onStop}
			class="group h-8 w-8 rounded-full p-0 hover:bg-destructive/10!"
		>
			<span class="sr-only">Stop</span>

			<Square
				class="h-8 w-8 fill-muted-foreground stroke-muted-foreground group-hover:fill-destructive group-hover:stroke-destructive hover:fill-destructive hover:stroke-destructive"
			/>
		</Button>
	{:else if shouldShowRecordButton}
		<ChatFormActionRecord {disabled} {hasAudioModality} {isLoading} {isRecording} {onMicClick} />
	{:else}
		<ChatFormActionSubmit
			canSend={canSend && hasModelSelected && isSelectedModelInCache}
			{disabled}
			{isLoading}
			tooltipLabel={submitTooltip}
			showErrorState={hasModelSelected && !isSelectedModelInCache}
		/>
	{/if}
</div>
