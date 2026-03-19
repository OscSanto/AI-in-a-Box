<script lang="ts">
	import { BookOpen, X, ExternalLink } from '@lucide/svelte';
	import { articles } from '$lib/stores/articlesStore.svelte';
	import { fly } from 'svelte/transition';

	interface Props {
		open: boolean;
		onClose: () => void;
	}

	let { open, onClose }: Props = $props();

	function truncateSentences(text: string, max = 3): string {
		const sentences = text.match(/[^.!?]*[.!?]+/g) ?? [];
		if (!sentences.length) return text.slice(0, 200) + (text.length > 200 ? '...' : '');
		return sentences.slice(0, max).join('') + (sentences.length > max ? '...' : '');
	}
</script>

{#if open}
	<aside
		transition:fly={{ x: 320, duration: 200 }}
		class="fixed top-0 right-0 z-[800] flex h-full w-72 flex-col border-l border-border bg-background shadow-lg"
	>
		<!-- Header -->
		<div class="flex items-center justify-between border-b border-border px-4 py-3">
			<div class="flex items-center gap-2 text-sm font-medium">
				<BookOpen class="h-4 w-4" />
				Sources
			</div>
			<button onclick={onClose} class="rounded p-1 hover:bg-muted">
				<X class="h-4 w-4" />
			</button>
		</div>

		<!-- Article cards -->
		<div class="flex-1 space-y-2 overflow-y-auto p-3">
			{#if articles().length === 0}
				<p class="px-1 pt-2 text-xs text-muted-foreground">
					No sources yet. Ask a question in Search mode.
				</p>
			{:else}
				{#each articles() as article (article.title)}
					<div class="space-y-1.5 rounded-lg border border-border bg-muted/30 p-3 text-xs">
						<a
							href={article.url}
							target="_blank"
							rel="noopener"
							class="flex items-start gap-1 font-medium leading-snug text-foreground hover:underline"
						>
							{article.title}
							<ExternalLink class="mt-0.5 h-3 w-3 shrink-0 text-muted-foreground" />
						</a>
						{#if article.snippet}
							<p class="leading-relaxed text-muted-foreground">
								{truncateSentences(article.snippet)}
							</p>
						{/if}
					</div>
				{/each}
			{/if}
		</div>
	</aside>
{/if}
