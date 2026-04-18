<script lang="ts">
	import {
		ChatAttachmentsList,
		ChatFormActions,
		ChatFormFileInputInvisible,
		ChatFormTextarea
	} from '$lib/components/app';
	import { INPUT_CLASSES } from '$lib/constants/css-classes';
	import { SETTING_CONFIG_DEFAULT } from '$lib/constants/settings-config';
	import { CLIPBOARD_CONTENT_QUOTE_PREFIX } from '$lib/constants/chat-form';
	import { KeyboardKey, MimeTypeText } from '$lib/enums';
	import { config } from '$lib/stores/settings.svelte';
	import {
		ragControls,
		resetBypassCache,
		resetLogLevel,
		resetRagMode,
		resetSelectedZim,
		resetThinking,
		toggleThinking,
		modelSupportsThinking,
		suggestRagCommand
	} from '$lib/stores/ragControls.svelte';
	import { modelOptions, selectedModelId, selectedModelName, singleModelName } from '$lib/stores/models.svelte';
	import { isRouterMode } from '$lib/stores/server.svelte';
	import { chatStore } from '$lib/stores/chat.svelte';
	import { activeMessages } from '$lib/stores/conversations.svelte';
	import { isIMEComposing, parseClipboardContent } from '$lib/utils';
	import {
		AudioRecorder,
		convertToWav,
		createAudioFile,
		isAudioRecordingSupported
	} from '$lib/utils/browser-only';
	import { onMount } from 'svelte';

	interface Props {
		// Data
		attachments?: DatabaseMessageExtra[];
		uploadedFiles?: ChatUploadedFile[];
		value?: string;

		// UI State
		class?: string;
		disabled?: boolean;
		isLoading?: boolean;
		placeholder?: string;

		// Event Handlers
		onAttachmentRemove?: (index: number) => void;
		onFilesAdd?: (files: File[]) => void;
		onStop?: () => void;
		onSubmit?: () => void;
		onSystemPromptClick?: (draft: { message: string; files: ChatUploadedFile[] }) => void;
		onUploadedFileRemove?: (fileId: string) => void;
		onValueChange?: (value: string) => void;
	}

	let {
		attachments = [],
		class: className = '',
		disabled = false,
		isLoading = false,
		placeholder = 'Type a message...',
		uploadedFiles = $bindable([]),
		value = $bindable(''),
		onAttachmentRemove,
		onFilesAdd,
		onStop,
		onSubmit,
		onSystemPromptClick,
		onUploadedFileRemove,
		onValueChange
	}: Props = $props();

	/**
	 *
	 *
	 * STATE
	 *
	 *
	 */

	// Component References
	let audioRecorder: AudioRecorder | undefined;
	let chatFormActionsRef: ChatFormActions | undefined = $state(undefined);
	let fileInputRef: ChatFormFileInputInvisible | undefined = $state(undefined);
	let textareaRef: ChatFormTextarea | undefined = $state(undefined);

	// Audio Recording State
	let isRecording = $state(false);
	let recordingSupported = $state(false);

	/**
	 *
	 *
	 * DERIVED STATE
	 *
	 *
	 */

	// Configuration
	let currentConfig = $derived(config());
	let currentRagControls = $derived(ragControls());
	let commandSuggestion = $derived(suggestRagCommand(value));
	let pasteLongTextToFileLength = $derived.by(() => {
		const n = Number(currentConfig.pasteLongTextToFileLen);
		return Number.isNaN(n) ? Number(SETTING_CONFIG_DEFAULT.pasteLongTextToFileLen) : n;
	});

	// Model Selection Logic
	let isRouter = $derived(isRouterMode());
	let conversationModel = $derived(
		chatStore.getConversationModel(activeMessages() as DatabaseMessage[])
	);
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

	// Thinking mode support — use selectedModelName (reactive) over singleModelName (stale props)
	let activeModelName = $derived(isRouter ? activeModelId : (selectedModelName() ?? singleModelName()));
	let supportsThinking = $derived(modelSupportsThinking(activeModelName));

	// Form Validation State
	let hasModelSelected = $derived(!isRouter || !!conversationModel || !!selectedModelId());
	let hasLoadingAttachments = $derived(uploadedFiles.some((f) => f.isLoading));
	let hasAttachments = $derived(
		(attachments && attachments.length > 0) || (uploadedFiles && uploadedFiles.length > 0)
	);
	let canSubmit = $derived(value.trim().length > 0 || hasAttachments);

	/**
	 *
	 *
	 * LIFECYCLE
	 *
	 *
	 */

	onMount(() => {
		recordingSupported = isAudioRecordingSupported();
		audioRecorder = new AudioRecorder();
	});

	/**
	 *
	 *
	 * PUBLIC API
	 *
	 *
	 */

	export function focus() {
		textareaRef?.focus();
	}

	export function resetTextareaHeight() {
		textareaRef?.resetHeight();
	}

	export function openModelSelector() {
		chatFormActionsRef?.openModelSelector();
	}

	/**
	 * Check if a model is selected, open selector if not
	 * @returns true if model is selected, false otherwise
	 */
	export function checkModelSelected(): boolean {
		if (!hasModelSelected) {
			chatFormActionsRef?.openModelSelector();
			return false;
		}
		return true;
	}

	/**
	 *
	 *
	 * EVENT HANDLERS - File Management
	 *
	 *
	 */

	function handleFileSelect(files: File[]) {
		onFilesAdd?.(files);
	}

	function handleFileUpload() {
		fileInputRef?.click();
	}

	function handleFileRemove(fileId: string) {
		if (fileId.startsWith('attachment-')) {
			const index = parseInt(fileId.replace('attachment-', ''), 10);
			if (!isNaN(index) && index >= 0 && index < attachments.length) {
				onAttachmentRemove?.(index);
			}
		} else {
			onUploadedFileRemove?.(fileId);
		}
	}

	/**
	 *
	 *
	 * EVENT HANDLERS - Input & Keyboard
	 *
	 *
	 */

	function handleKeydown(event: KeyboardEvent) {
		if (event.key === KeyboardKey.ENTER && !event.shiftKey && !isIMEComposing(event)) {
			event.preventDefault();

			if (!canSubmit || disabled || isLoading || hasLoadingAttachments) return;

			onSubmit?.();
		}
	}

	function handlePaste(event: ClipboardEvent) {
		if (!event.clipboardData) return;

		const files = Array.from(event.clipboardData.items)
			.filter((item) => item.kind === 'file')
			.map((item) => item.getAsFile())
			.filter((file): file is File => file !== null);

		if (files.length > 0) {
			event.preventDefault();
			onFilesAdd?.(files);
			return;
		}

		const text = event.clipboardData.getData(MimeTypeText.PLAIN);

		if (text.startsWith(CLIPBOARD_CONTENT_QUOTE_PREFIX)) {
			const parsed = parseClipboardContent(text);

			if (parsed.textAttachments.length > 0) {
				event.preventDefault();
				value = parsed.message;
				onValueChange?.(parsed.message);

				// Handle text attachments as files
				if (parsed.textAttachments.length > 0) {
					const attachmentFiles = parsed.textAttachments.map(
						(att) =>
							new File([att.content], att.name, {
								type: MimeTypeText.PLAIN
							})
					);
					onFilesAdd?.(attachmentFiles);
				}

				setTimeout(() => {
					textareaRef?.focus();
				}, 10);

				return;
			}
		}

		if (
			text.length > 0 &&
			pasteLongTextToFileLength > 0 &&
			text.length > pasteLongTextToFileLength
		) {
			event.preventDefault();

			const textFile = new File([text], 'Pasted', {
				type: MimeTypeText.PLAIN
			});

			onFilesAdd?.([textFile]);
		}
	}

	/**
	 *
	 *
	 * EVENT HANDLERS - Audio Recording
	 *
	 *
	 */

	async function handleMicClick() {
		if (!audioRecorder || !recordingSupported) {
			console.warn('Audio recording not supported');
			return;
		}

		if (isRecording) {
			try {
				const audioBlob = await audioRecorder.stopRecording();
				const wavBlob = await convertToWav(audioBlob);
				const audioFile = createAudioFile(wavBlob);

				onFilesAdd?.([audioFile]);
				isRecording = false;
			} catch (error) {
				console.error('Failed to stop recording:', error);
				isRecording = false;
			}
		} else {
			try {
				await audioRecorder.startRecording();
				isRecording = true;
			} catch (error) {
				console.error('Failed to start recording:', error);
			}
		}
	}
</script>

<ChatFormFileInputInvisible bind:this={fileInputRef} onFileSelect={handleFileSelect} />

<form
	class="relative {className}"
	onsubmit={(e) => {
		e.preventDefault();
		if (!canSubmit || disabled || isLoading || hasLoadingAttachments) return;
		onSubmit?.();
	}}
>
	<div
		class="{INPUT_CLASSES} overflow-hidden rounded-3xl backdrop-blur-md {disabled
			? 'cursor-not-allowed opacity-60'
			: ''}"
		data-slot="input-area"
	>
		<ChatAttachmentsList
			{attachments}
			bind:uploadedFiles
			onFileRemove={handleFileRemove}
			limitToSingleRow
			class="py-5"
			style="scroll-padding: 1rem;"
			activeModelId={activeModelId ?? undefined}
		/>

		<div
			class="flex-column relative min-h-[48px] items-center rounded-3xl py-2 pb-2.25 shadow-sm transition-all focus-within:shadow-md md:!py-3"
			onpaste={handlePaste}
		>
			<div class="flex flex-wrap gap-1.5 px-5 pb-2 text-xs">
				{#if currentRagControls.mode !== 'balanced'}
					<button
						type="button"
						class="rounded-full border border-border bg-muted/70 px-2 py-0.5 text-muted-foreground transition hover:text-foreground"
						onclick={resetRagMode}
					>
						Mode: {currentRagControls.mode} x
					</button>
				{/if}
				{#if currentRagControls.selectedZim !== 'all'}
					<button
						type="button"
						class="rounded-full border border-border bg-muted/70 px-2 py-0.5 text-muted-foreground transition hover:text-foreground"
						onclick={resetSelectedZim}
					>
						ZIM: {currentRagControls.selectedZim} x
					</button>
				{/if}
				{#if currentRagControls.bypassCache}
					<button
						type="button"
						class="rounded-full border border-border bg-muted/70 px-2 py-0.5 text-muted-foreground transition hover:text-foreground"
						onclick={resetBypassCache}
					>
						New answers x
					</button>
				{/if}
				{#if currentRagControls.logLevel !== 'off'}
					<button
						type="button"
						class="rounded-full border border-border bg-muted/70 px-2 py-0.5 text-muted-foreground transition hover:text-foreground"
						onclick={resetLogLevel}
					>
						Logs: {currentRagControls.logLevel} x
					</button>
				{/if}
				{#if supportsThinking}
					<button
						type="button"
						class="rounded-full border px-2 py-0.5 text-xs transition {currentRagControls.thinking
							? 'border-violet-400 bg-violet-500/10 text-violet-500 hover:bg-violet-500/20'
							: 'border-border bg-muted/70 text-muted-foreground hover:text-foreground'}"
						onclick={toggleThinking}
						title={currentRagControls.thinking ? 'Thinking on — click to disable' : 'Thinking off — click to enable'}
					>
						{currentRagControls.thinking ? '🧠 Think ✕' : '🧠 Think'}
					</button>
				{/if}
				{#if commandSuggestion}
					<span class="rounded-full border border-dashed border-border px-2 py-0.5 text-muted-foreground">
						Tab: {commandSuggestion}
					</span>
				{/if}
			</div>

			<ChatFormTextarea
				class="px-5 py-1.5 md:pt-0"
				bind:this={textareaRef}
				bind:value
				onKeydown={handleKeydown}
				onInput={() => {
					onValueChange?.(value);
				}}
				{disabled}
				{placeholder}
			/>

			<ChatFormActions
				class="px-3"
				bind:this={chatFormActionsRef}
				canSend={canSubmit}
				hasText={value.trim().length > 0}
				{disabled}
				{isLoading}
				{isRecording}
				{uploadedFiles}
				onFileUpload={handleFileUpload}
				onMicClick={handleMicClick}
				{onStop}
				onSystemPromptClick={() => onSystemPromptClick?.({ message: value, files: uploadedFiles })}
			/>
		</div>
	</div>
</form>
