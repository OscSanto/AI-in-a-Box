<script lang="ts">
	import { Settings, BookOpen, AlertTriangle } from '@lucide/svelte';
	import { DialogChatSettings } from '$lib/components/app';
	import { ModelsSelector } from '$lib/components/app';
	import { Button } from '$lib/components/ui/button';
	import { useSidebar } from '$lib/components/ui/sidebar';
	import { inferenceBackendsStore } from '$lib/stores/inferenceBackends.svelte';

	interface Props {
		onToggleSources?: () => void;
	}

	let { onToggleSources }: Props = $props();

	const sidebar = useSidebar();

	let settingsOpen = $state(false);
	let backendsSettingsOpen = $state(false);

	function toggleSettings() {
		settingsOpen = true;
	}

	function openBackendsSettings() {
		backendsSettingsOpen = true;
	}

	const activeBackend = $derived(inferenceBackendsStore.activeBackendName);
	const isPiLocal = $derived(activeBackend === 'Pi Local');
</script>

<header
	class="pointer-events-none fixed top-0 right-0 left-0 z-50 flex items-center justify-between p-4 duration-200 ease-linear {sidebar.open
		? 'md:left-[var(--sidebar-width)]'
		: ''}"
>
	<div class="pointer-events-auto flex items-center gap-2 {!sidebar.open ? 'pl-10' : ''}">
		<ModelsSelector forceForegroundText={true} />

		{#if activeBackend && isPiLocal}
			<button
				onclick={openBackendsSettings}
				class="flex items-center gap-1 rounded-full bg-amber-500/15 px-2 py-0.5 text-xs font-medium text-amber-600 backdrop-blur-lg hover:bg-amber-500/25 dark:text-amber-400"
				title="Running on Pi — click to configure remote backends"
			>
				<AlertTriangle class="h-3 w-3" />
				Pi Local
			</button>
		{:else if activeBackend && activeBackend !== 'Pi Local'}
			<span
				class="rounded-full bg-green-500/15 px-2 py-0.5 text-xs font-medium text-green-600 backdrop-blur-lg dark:text-green-400"
				title="Using remote backend: {activeBackend}"
			>
				{activeBackend}
			</span>
		{/if}
	</div>

	<div class="pointer-events-auto flex items-center space-x-2">
		<Button
			variant="ghost"
			size="icon"
			onclick={onToggleSources}
			class="rounded-full backdrop-blur-lg"
		>
			<BookOpen class="h-4 w-4" />
		</Button>
		<Button
			variant="ghost"
			size="icon"
			onclick={toggleSettings}
			class="rounded-full backdrop-blur-lg"
		>
			<Settings class="h-4 w-4" />
		</Button>
	</div>
</header>

<DialogChatSettings open={settingsOpen} onOpenChange={(open) => (settingsOpen = open)} />
<DialogChatSettings
	open={backendsSettingsOpen}
	onOpenChange={(open) => (backendsSettingsOpen = open)}
	initialSection="Backends"
/>
