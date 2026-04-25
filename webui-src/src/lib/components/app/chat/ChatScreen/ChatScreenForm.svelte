<script lang="ts">
	import { afterNavigate } from '$app/navigation';
	import { ChatFormHelperText, ChatForm } from '$lib/components/app';
	import { parseRagCommand, ragControls, resetForkActive } from '$lib/stores/ragControls.svelte';
	import { GitFork, X } from '@lucide/svelte';
	import { onMount } from 'svelte';

	interface Props {
		class?: string;
		disabled?: boolean;
		initialMessage?: string;
		isLoading?: boolean;
		onFileRemove?: (fileId: string) => void;
		onFileUpload?: (files: File[]) => void;
		onSend?: (message: string, files?: ChatUploadedFile[]) => Promise<boolean>;
		onStop?: () => void;
		onSystemPromptAdd?: (draft: { message: string; files: ChatUploadedFile[] }) => void;
		showHelperText?: boolean;
		uploadedFiles?: ChatUploadedFile[];
	}

	let {
		class: className,
		disabled = false,
		initialMessage = '',
		isLoading = false,
		onFileRemove,
		onFileUpload,
		onSend,
		onStop,
		onSystemPromptAdd,
		showHelperText = true,
		uploadedFiles = $bindable([])
	}: Props = $props();

	let chatFormRef: ChatForm | undefined = $state(undefined);
	let message = $derived(initialMessage);
	let previousIsLoading = $derived(isLoading);
	let previousInitialMessage = $derived(initialMessage);

	// Sync message when initialMessage prop changes (e.g., after draft restoration)
	$effect(() => {
		if (initialMessage !== previousInitialMessage) {
			message = initialMessage;
			previousInitialMessage = initialMessage;
		}
	});

	function handleSystemPromptClick() {
		onSystemPromptAdd?.({ message, files: uploadedFiles });
	}

	let hasLoadingAttachments = $derived(uploadedFiles.some((f) => f.isLoading));
	let isForkActive = $derived(ragControls().forkActive);

	async function handleSubmit() {
		if (
			(!message.trim() && uploadedFiles.length === 0) ||
			disabled ||
			isLoading ||
			hasLoadingAttachments
		)
			return;

		const messageToSend = message.trim();
		const command = parseRagCommand(messageToSend);
		if (command.handled) {
			message = command.remaining;
			chatFormRef?.resetTextareaHeight();
			if (command.message) window.alert(command.message);
			if (command.remaining) {
				return handleSubmit();
			}
			return;
		}

		if (!chatFormRef?.checkModelSelected()) return;

		const filesToSend = [...uploadedFiles];

		message = '';
		uploadedFiles = [];

		chatFormRef?.resetTextareaHeight();

		const success = await onSend?.(messageToSend, filesToSend);

		if (!success) {
			message = messageToSend;
			uploadedFiles = filesToSend;
		}
	}

	function handleFilesAdd(files: File[]) {
		onFileUpload?.(files);
	}

	function handleUploadedFileRemove(fileId: string) {
		onFileRemove?.(fileId);
	}

	onMount(() => {
		setTimeout(() => chatFormRef?.focus(), 10);
	});

	afterNavigate(() => {
		setTimeout(() => chatFormRef?.focus(), 10);
	});

	$effect(() => {
		if (previousIsLoading && !isLoading) {
			setTimeout(() => chatFormRef?.focus(), 10);
		}

		previousIsLoading = isLoading;
	});
</script>

<div class="relative mx-auto max-w-[48rem]">
	{#if isForkActive}
		<div class="mb-1 flex items-center gap-1.5 px-1">
			<GitFork class="h-3 w-3 text-primary" />
			<span class="text-xs text-primary">Extended conversation</span>
			<button
				class="ml-auto flex h-4 w-4 items-center justify-center rounded hover:bg-muted"
				onclick={resetForkActive}
				aria-label="Disable extended conversation"
			>
				<X class="h-3 w-3 text-muted-foreground" />
			</button>
		</div>
	{/if}
	<ChatForm
		bind:this={chatFormRef}
		bind:value={message}
		bind:uploadedFiles
		class={className}
		{disabled}
		{isLoading}
		onFilesAdd={handleFilesAdd}
		{onStop}
		onSubmit={handleSubmit}
		onSystemPromptClick={handleSystemPromptClick}
		onUploadedFileRemove={handleUploadedFileRemove}
	/>
</div>

<ChatFormHelperText show={showHelperText} />
