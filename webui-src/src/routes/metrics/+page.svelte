<script lang="ts">
	import { onMount, onDestroy } from 'svelte';

	type TempSensor = { chip: string; label: string; current: number; high: number | null; critical: number | null };
	type NetIface   = { iface: string; ipv4: string | null; sent_mb: number; recv_mb: number; errors_in: number; errors_out: number };
	type GpuEntry   = { name: string; vram_used_mb: number; vram_total_mb: number; util_pct: number; temp_c: number };

	type Metrics = {
		uptime_s:     number;
		cpu: {
			per_core_pct:   number[];
			avg_pct:        number;
			count_logical:  number;
			count_physical: number;
			freq_mhz:       number | null;
			freq_max_mhz:   number | null;
		};
		memory: {
			ram_used_mb:   number;
			ram_total_mb:  number;
			ram_pct:       number;
			ram_avail_mb:  number;
			swap_used_mb:  number;
			swap_total_mb: number;
			swap_pct:      number;
		};
		disk: {
			path:     string;
			used_gb:  number;
			total_gb: number;
			free_gb:  number;
			pct:      number;
			read_mb:  number | null;
			write_mb: number | null;
		};
		temperatures: TempSensor[];
		network:      NetIface[];
		process: {
			pid:        number;
			rss_mb:     number;
			vms_mb:     number;
			cpu_pct:    number;
			threads:    number;
			open_files: number;
		};
		gpu: GpuEntry[] | null;
	};

	let metrics: Metrics | null = null;
	let error = '';
	let interval: ReturnType<typeof setInterval>;
	const REFRESH_MS = 2000;

	function formatUptime(s: number): string {
		const h = Math.floor(s / 3600);
		const m = Math.floor((s % 3600) / 60);
		return h > 0 ? `${h}h ${m}m` : `${m}m`;
	}

	function barColor(pct: number): string {
		if (pct > 85) return '#ff4455';
		if (pct > 60) return '#ffaa33';
		return '#44cc88';
	}

	function tempColor(c: number): string {
		if (c > 80) return '#ff4455';
		if (c > 60) return '#ffaa33';
		return '#44cc88';
	}

	async function refresh() {
		try {
			const res = await fetch('/api/metrics');
			if (!res.ok) throw new Error(`HTTP ${res.status}`);
			metrics = await res.json();
			error = '';
		} catch (e: any) {
			error = String(e);
		}
	}

	onMount(() => {
		refresh();
		interval = setInterval(refresh, REFRESH_MS);
	});

	onDestroy(() => clearInterval(interval));

	// Derive CPU temp from sensors (pick the highest "package" or first CPU sensor)
	function cpuTemp(sensors: TempSensor[]): number | null {
		if (!sensors?.length) return null;
		const pkg = sensors.find(s => s.label.toLowerCase().includes('package') || s.label.toLowerCase().includes('tdie'));
		return pkg ? pkg.current : sensors[0].current;
	}
</script>

<div class="page">
	<div class="header">
		<a href="/" class="back">← Chat</a>
		<h1>System Metrics</h1>
		<span class="refresh-badge">↻ {REFRESH_MS / 1000}s</span>
	</div>

	{#if error}
		<div class="error">{error}</div>
	{:else if !metrics}
		<div class="loading">Loading…</div>
	{:else}
		<div class="grid">

			<!-- CPU -->
			<div class="card">
				<div class="card-title">CPU</div>
				<div class="big-num">{metrics.cpu.avg_pct}<span class="unit">%</span></div>
				<div class="bar-track">
					<div class="bar-fill" style="width:{metrics.cpu.avg_pct}%; background:{barColor(metrics.cpu.avg_pct)}"></div>
				</div>
				<div class="sub-row">
					{#each metrics.cpu.per_core_pct as pct, i}
						<div class="core-bar" title="Core {i}: {Math.round(pct)}%">
							<div class="core-fill" style="height:{pct}%; background:{barColor(pct)}"></div>
						</div>
					{/each}
				</div>
				<div class="label">
					{metrics.cpu.count_physical}P / {metrics.cpu.count_logical}L
					{#if metrics.cpu.freq_mhz} · {metrics.cpu.freq_mhz} MHz{/if}
					{#if metrics.cpu.freq_max_mhz} / {metrics.cpu.freq_max_mhz} max{/if}
				</div>
			</div>

			<!-- RAM -->
			<div class="card">
				<div class="card-title">RAM</div>
				<div class="big-num">{metrics.memory.ram_pct}<span class="unit">%</span></div>
				<div class="bar-track">
					<div class="bar-fill" style="width:{metrics.memory.ram_pct}%; background:{barColor(metrics.memory.ram_pct)}"></div>
				</div>
				<div class="label">{metrics.memory.ram_used_mb} MB / {metrics.memory.ram_total_mb} MB</div>
				{#if metrics.memory.swap_total_mb > 0}
					<div class="label mt4">Swap {metrics.memory.swap_used_mb} / {metrics.memory.swap_total_mb} MB ({metrics.memory.swap_pct}%)</div>
				{/if}
			</div>

			<!-- CPU Temp (from sensors) -->
			<div class="card">
				<div class="card-title">CPU Temp</div>
				{#if cpuTemp(metrics.temperatures) !== null}
					{@const t = cpuTemp(metrics.temperatures)!}
					<div class="big-num" style="color:{tempColor(t)}">{t}<span class="unit">°C</span></div>
					<div class="bar-track">
						<div class="bar-fill" style="width:{Math.min(t, 100)}%; background:{tempColor(t)}"></div>
					</div>
				{:else}
					<div class="na">N/A</div>
				{/if}
			</div>

			<!-- Disk -->
			<div class="card">
				<div class="card-title">Disk</div>
				<div class="big-num">{metrics.disk.pct}<span class="unit">%</span></div>
				<div class="bar-track">
					<div class="bar-fill" style="width:{metrics.disk.pct}%; background:{barColor(metrics.disk.pct)}"></div>
				</div>
				<div class="label">{metrics.disk.used_gb} GB / {metrics.disk.total_gb} GB</div>
				{#if metrics.disk.read_mb !== null}
					<div class="label mt4">R {metrics.disk.read_mb} MB · W {metrics.disk.write_mb} MB</div>
				{/if}
			</div>

			<!-- Process (this server) -->
			<div class="card">
				<div class="card-title">Server Process</div>
				<div class="big-num small">{metrics.process.rss_mb}<span class="unit">MB</span></div>
				<div class="label">RSS · VMS {metrics.process.vms_mb} MB</div>
				<div class="label mt4">CPU {metrics.process.cpu_pct}% · {metrics.process.threads} threads</div>
				<div class="label">{metrics.process.open_files} open files</div>
			</div>

			<!-- Uptime -->
			<div class="card">
				<div class="card-title">Uptime</div>
				<div class="big-num small">{formatUptime(metrics.uptime_s)}</div>
			</div>

			<!-- GPU (if present) -->
			{#if metrics.gpu?.length}
				{#each metrics.gpu as g}
					<div class="card">
						<div class="card-title">GPU</div>
						<div class="big-num">{g.util_pct}<span class="unit">%</span></div>
						<div class="bar-track">
							<div class="bar-fill" style="width:{g.util_pct}%; background:{barColor(g.util_pct)}"></div>
						</div>
						<div class="label">{g.name}</div>
						<div class="label mt4">VRAM {g.vram_used_mb} / {g.vram_total_mb} MB · {g.temp_c}°C</div>
					</div>
				{/each}
			{/if}

			<!-- Per-core detail -->
			<div class="card wide">
				<div class="card-title">Per-Core CPU %</div>
				<div class="cores-detail">
					{#each metrics.cpu.per_core_pct as pct, i}
						<div class="core-row">
							<span class="core-label">C{i}</span>
							<div class="bar-track flex1">
								<div class="bar-fill" style="width:{pct}%; background:{barColor(pct)}"></div>
							</div>
							<span class="core-val">{Math.round(pct)}%</span>
						</div>
					{/each}
				</div>
			</div>

			<!-- All temperature sensors -->
			{#if metrics.temperatures.length > 1}
				<div class="card wide">
					<div class="card-title">Sensors</div>
					<div class="sensor-grid">
						{#each metrics.temperatures as s}
							<div class="sensor-row">
								<span class="sensor-label">{s.label}</span>
								<span class="sensor-val" style="color:{tempColor(s.current)}">{s.current}°C</span>
								{#if s.high}<span class="sensor-limit">/ {s.high}°</span>{/if}
							</div>
						{/each}
					</div>
				</div>
			{/if}

			<!-- Network -->
			{#if metrics.network.length}
				<div class="card wide">
					<div class="card-title">Network</div>
					<div class="net-table">
						{#each metrics.network as iface}
							<div class="net-row">
								<span class="net-iface">{iface.iface}</span>
								{#if iface.ipv4}<span class="net-ip">{iface.ipv4}</span>{/if}
								<span class="net-stat">↑ {iface.sent_mb} MB</span>
								<span class="net-stat">↓ {iface.recv_mb} MB</span>
								{#if iface.errors_in || iface.errors_out}
									<span class="net-err">err {iface.errors_in}/{iface.errors_out}</span>
								{/if}
							</div>
						{/each}
					</div>
				</div>
			{/if}

		</div>
	{/if}
</div>

<style>
	.page {
		min-height: 100vh;
		background: var(--background);
		color: var(--foreground);
		font-family: 'JetBrains Mono', 'Fira Code', monospace, sans-serif;
		padding: 24px;
	}

	.header {
		display: flex;
		align-items: center;
		gap: 16px;
		margin-bottom: 28px;
	}

	h1 { font-size: 18px; font-weight: 600; color: var(--foreground); margin: 0; }

	.back {
		color: var(--muted-foreground);
		text-decoration: none;
		font-size: 13px;
		padding: 5px 10px;
		border: 1px solid var(--border);
		border-radius: 6px;
	}
	.back:hover { color: var(--foreground); border-color: var(--ring); }

	.refresh-badge {
		margin-left: auto;
		font-size: 11px;
		color: #44cc88;
		background: rgba(68,204,136,0.1);
		padding: 3px 8px;
		border-radius: 4px;
		border: 1px solid rgba(68,204,136,0.2);
	}

	.grid {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
		gap: 16px;
	}

	.card {
		background: var(--card);
		border: 1px solid var(--border);
		border-radius: 10px;
		padding: 18px;
	}
	.card.wide { grid-column: span 2; }

	.card-title {
		font-size: 10px;
		text-transform: uppercase;
		letter-spacing: 0.1em;
		color: var(--muted-foreground);
		margin-bottom: 10px;
	}

	.big-num {
		font-size: 38px;
		font-weight: 700;
		color: var(--foreground);
		line-height: 1;
		margin-bottom: 10px;
	}
	.big-num.small { font-size: 28px; }

	.unit { font-size: 16px; color: var(--muted-foreground); margin-left: 2px; }

	.bar-track {
		height: 6px;
		background: var(--muted);
		border-radius: 3px;
		overflow: hidden;
		margin-bottom: 8px;
	}
	.bar-track.flex1 { flex: 1; margin-bottom: 0; }

	.bar-fill {
		height: 100%;
		border-radius: 3px;
		transition: width 0.4s ease;
	}

	.label { font-size: 11px; color: var(--muted-foreground); margin-top: 4px; }
	.mt4   { margin-top: 4px; }

	.na { font-size: 24px; color: var(--muted-foreground); margin: 8px 0; }

	/* CPU mini bars */
	.sub-row {
		display: flex;
		gap: 3px;
		align-items: flex-end;
		height: 30px;
		margin: 6px 0;
	}
	.core-bar {
		flex: 1;
		height: 100%;
		background: var(--muted);
		border-radius: 2px;
		display: flex;
		align-items: flex-end;
		overflow: hidden;
	}
	.core-fill { width: 100%; border-radius: 2px; transition: height 0.4s ease; }

	/* Per-core detail */
	.cores-detail { display: flex; flex-direction: column; gap: 6px; }
	.core-row { display: flex; align-items: center; gap: 8px; }
	.core-label { font-size: 10px; color: var(--muted-foreground); width: 18px; flex-shrink: 0; }
	.core-val { font-size: 11px; color: var(--muted-foreground); width: 32px; text-align: right; flex-shrink: 0; }

	/* Sensors */
	.sensor-grid { display: flex; flex-direction: column; gap: 5px; }
	.sensor-row { display: flex; align-items: baseline; gap: 8px; }
	.sensor-label { font-size: 11px; color: var(--muted-foreground); flex: 1; }
	.sensor-val { font-size: 13px; font-weight: 600; }
	.sensor-limit { font-size: 10px; color: var(--muted-foreground); }

	/* Network */
	.net-table { display: flex; flex-direction: column; gap: 6px; }
	.net-row { display: flex; align-items: baseline; gap: 12px; font-size: 11px; flex-wrap: wrap; }
	.net-iface { color: var(--foreground); font-weight: 600; min-width: 60px; }
	.net-ip { color: var(--muted-foreground); }
	.net-stat { color: var(--muted-foreground); }
	.net-err { color: #ff6644; }

	.loading, .error {
		color: var(--muted-foreground);
		font-size: 13px;
		padding: 40px;
		text-align: center;
	}
	.error { color: #ff6666; }
</style>
