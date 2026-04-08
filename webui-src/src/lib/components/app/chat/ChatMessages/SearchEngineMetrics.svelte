<script lang="ts">
	interface Props {
		metrics: Record<string, unknown>;
	}

	const { metrics }: Props = $props();

	const fmt = (v: unknown, unit: string): string =>
		typeof v === 'number' ? v.toFixed(2) + unit : '—';

	const fmtInt = (v: unknown, unit: string): string =>
		typeof v === 'number' ? Math.round(v) + unit : '—';

	const items = [
		{ label: 'Prefill',  value: fmt(metrics.prefill_s, 's') },
		{ label: 'Generate', value: fmt(metrics.gen_s, 's') },
		{ label: 'Speed',    value: typeof metrics.gen_tok_s === 'number' ? Math.round(metrics.gen_tok_s) + ' tok/s' : '—' },
		{ label: 'Context',  value: fmtInt(metrics.prompt_tokens, ' tok') },
		{ label: 'Output',   value: fmtInt(metrics.gen_tokens, ' tok') },
	];

	const wasCold     = metrics.was_cold     === true;
	const hitLimit    = metrics.hit_token_limit === true;
</script>

<div class="mt-3 flex flex-wrap items-center gap-x-5 gap-y-1.5 border-t border-border pt-2.5">
	{#each items as m (m.label)}
		<span class="flex flex-col gap-0.5">
			<span class="text-[10px] font-medium uppercase tracking-wide text-muted-foreground">
				{m.label}
			</span>
			<span class="font-mono text-xs text-foreground/70">{m.value}</span>
		</span>
	{/each}

	<span
		class="self-center rounded-full px-2 py-0.5 text-[11px] font-semibold {wasCold
			? 'bg-orange-100 text-orange-700 dark:bg-orange-900/30 dark:text-orange-400'
			: 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400'}"
	>
		● {wasCold ? 'Cold' : 'Warm'}
	</span>

	{#if hitLimit}
		<span
			class="self-center rounded-full bg-yellow-100 px-2 py-0.5 text-[11px] font-semibold text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400"
		>
			Truncated
		</span>
	{/if}
</div>
