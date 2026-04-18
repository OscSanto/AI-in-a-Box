<script lang="ts">
	import { completeRagCommand } from '$lib/stores/ragControls.svelte';
	import { autoResizeTextarea } from '$lib/utils';
	import { onMount } from 'svelte';

	interface Props {
		class?: string;
		disabled?: boolean;
		onInput?: () => void;
		onKeydown?: (event: KeyboardEvent) => void;
		onPaste?: (event: ClipboardEvent) => void;
		placeholder?: string;
		value?: string;
	}

	let {
		class: className = '',
		disabled = false,
		onInput,
		onKeydown,
		onPaste,
		placeholder = 'Ask anything...',
		value = $bindable('')
	}: Props = $props();

	let textareaElement: HTMLTextAreaElement | undefined;

	onMount(() => {
		if (textareaElement) {
			textareaElement.focus();
		}
	});

	// Expose the textarea element for external access
	export function getElement() {
		return textareaElement;
	}

	export function focus() {
		textareaElement?.focus();
	}

	export function resetHeight() {
		if (textareaElement) {
			textareaElement.style.height = '1rem';
		}
	}

	function handleKeydown(event: KeyboardEvent) {
		if (event.key === 'Tab' && textareaElement) {
			const completion = completeRagCommand(value, textareaElement.selectionStart);
			if (completion) {
				event.preventDefault();
				value = completion;
				queueMicrotask(() => {
					if (!textareaElement) return;
					textareaElement.selectionStart = completion.length;
					textareaElement.selectionEnd = completion.length;
					autoResizeTextarea(textareaElement);
				});
				onInput?.();
				return;
			}
		}

		onKeydown?.(event);
	}
</script>

<div class="flex-1 {className}">
	<textarea
		bind:this={textareaElement}
		bind:value
		class="text-md max-h-32 min-h-12 w-full resize-none border-0 bg-transparent p-0 leading-6 outline-none placeholder:text-muted-foreground focus-visible:ring-0 focus-visible:ring-offset-0"
		class:cursor-not-allowed={disabled}
		{disabled}
		onkeydown={handleKeydown}
		oninput={(event) => {
			autoResizeTextarea(event.currentTarget);
			onInput?.();
		}}
		onpaste={onPaste}
		{placeholder}
	></textarea>
</div>
