<script lang="ts">
	import IconesCSVFile from '../../components/icons/IconesCSVFile.svelte';
	import Histogram from "../../components/ui/Histogram.svelte";
	import TwoHistogram from "../../components/ui/TwoHistogram.svelte";
	import ScatterPlot from '../../components/ui/ScatterPlot.svelte';
	import BoxPlot from '../../components/ui/BoxPlot.svelte';
	import Sunburst from '../../components/ui/Sunburst.svelte';
	import {onMount} from "svelte";

	let statistics = $state({})
	let loading = $state(false)


	function formatNumber(num: number) {
		if (num === null || num === undefined) return 'N/A';
		return num.toLocaleString('en-US', { maximumFractionDigits: 2 });
	}

	onMount(async () => {
		try {
			loading = true
			const response = await fetch("http://localhost:8000/statistics");

			if (!response.ok) {
				throw new Error(`HTTP error! status: ${response.status}`);
			}

			statistics = await response.json();
			loading = false;
		} catch (err) {
			// error = err.message;
			loading = false;
			console.error('Error fetching statistics:', err);
		}
	});

</script>

{#if loading || Object.keys(statistics).length == 0}
	<main class="container">
		<section class="mt-32">
			<h1 class="text-3xl font-semibold text-text-primary">Loading...</h1>
		</section>
	</main>
{:else}
<main class="container">
	<section class="mt-32">
		<h1 class="text-3xl font-semibold text-text-primary">Dataset Statistics</h1>
		<h2 class="text-text-secondary">Interactive view of the dataset our model is trained on</h2>
	</section>

	<section class="mt-10 grid grid-cols-4 gap-16">
	<!-- Number of Data Packets -->
	<div class="rounded-lg border border-dashed border-border-primary bg-primary px-4 py-3">
		<h3 class="text-sm uppercase font-semibold tracking-wide text-text-primary/80">
		Number of Data Packets
		</h3>
		<p class="mt-2 text-4xl font-extrabold text-sky-400">{formatNumber(statistics.number_of_data_points)}</p>
	</div>

	<!-- Data Size -->
	<div class="rounded-lg border border-dashed border-border-primary bg-primary px-4 py-3">
		<h3 class="text-sm uppercase font-semibold tracking-wide text-text-primary/80">
		Data Size (MB)
		</h3>
		<p class="mt-2 text-4xl font-extrabold text-amber-400">{Math.floor(statistics.data_size.megabytes)}MB</p>
	</div>

	<!-- Anomaly Count -->
	<div class="rounded-lg border border-dashed border-border-primary bg-primary px-4 py-3">
		<h3 class="text-sm uppercase font-semibold tracking-wide text-text-primary/80">
		Anomaly Count
		</h3>
		<p class="mt-2 text-4xl font-extrabold text-rose-400">{formatNumber(statistics.anomaly_count)}</p>
	</div>

	<!-- Benign Count -->
	<div class="rounded-lg border border-dashed border-border-primary bg-primary px-4 py-3">
		<h3 class="text-sm uppercase font-semibold tracking-wide text-text-primary/80">
		Benign Count
		</h3>
		<p class="mt-2 text-4xl font-extrabold text-emerald-400">{formatNumber(statistics.benign_count)}</p>
	</div>
	</section>

	<!-- Histogram only (full width, 1 column) -->
	<section class="mt-10 grid grid-cols-1 gap-6">
	<div class="rounded-lg border border-dashed border-border-primary bg-primary px-4 py-3">
		<Sunburst class="min-h-96 w-full" data={statistics.sunburst_data} />
	</div>
</section>

<!-- Sunburst and Scatter Plot (2 columns) -->
<section class="mt-10 grid grid-cols-2 gap-6">
	<div class="rounded-lg border border-dashed border-border-primary bg-primary px-4 py-3">
		<BoxPlot data={statistics.box_plot_data} class="min-h-96 w-full" />
	</div>
	
	<div class="rounded-lg border border-dashed border-border-primary bg-primary px-4 py-3">
		<ScatterPlot data={statistics.scatter_plot_data} class="min-h-96 w-full" />
	</div>
</section>

<!-- Box Plot only (full width, 1 column) -->
<!--<section class="mt-10 grid grid-cols-1 gap-6">-->
<!--	<div class="rounded-lg border border-dashed border-border-primary bg-primary px-4 py-3">-->
<!--		<twohistogram data={statistics.histogram_data} class="min-h-96 w-full" />-->
<!--	</div>-->
<!--</section>-->

</main>
{/if}