<script>
	import IconesCSVFile from '../../components/icons/IconesCSVFile.svelte';
	import Histogram from "../../components/ui/Histogram.svelte";
	import TwoHistogram from "../../components/ui/TwoHistogram.svelte";
	import ScatterPlot from '../../components/ui/ScatterPlot.svelte';
	import BoxPlot from '../../components/ui/BoxPlot.svelte';
	import Sunburst from '../../components/ui/Sunburst.svelte';
	export let packets = 33716;
	export let dataSizeMB = 382;         // already in MB
	export let anomalyCount = 17171;
	export let benignCount = 16545;
</script>

<main class="container">
	<section class="mt-32">
		<h1 class="text-3xl font-semibold text-text-primary">Model Results</h1>
		<h2 class="text-text-secondary">Completed at 4:10pm | Took 60s</h2>
	</section>

	<section class="mt-10 grid grid-cols-4 gap-16">
	<!-- Number of Data Packets -->
	<div class="rounded-lg border border-dashed border-border-primary bg-primary px-4 py-3">
		<h3 class="text-sm uppercase font-semibold tracking-wide text-text-primary/80">
		Number of Data Packets
		</h3>
		<p class="mt-2 text-4xl font-extrabold text-sky-400">33,716</p>
	</div>

	<!-- Data Size -->
	<div class="rounded-lg border border-dashed border-border-primary bg-primary px-4 py-3">
		<h3 class="text-sm uppercase font-semibold tracking-wide text-text-primary/80">
		Data Size (MB)
		</h3>
		<p class="mt-2 text-4xl font-extrabold text-amber-400">382</p>
	</div>

	<!-- Anomaly Count -->
	<div class="rounded-lg border border-dashed border-border-primary bg-primary px-4 py-3">
		<h3 class="text-sm uppercase font-semibold tracking-wide text-text-primary/80">
		Anomaly Count
		</h3>
		<p class="mt-2 text-4xl font-extrabold text-rose-400">17,171</p>
	</div>

	<!-- Benign Count -->
	<div class="rounded-lg border border-dashed border-border-primary bg-primary px-4 py-3">
		<h3 class="text-sm uppercase font-semibold tracking-wide text-text-primary/80">
		Benign Count
		</h3>
		<p class="mt-2 text-4xl font-extrabold text-emerald-400">16,545</p>
	</div>
	</section>

	<!-- Histogram only (full width, 1 column) -->
	<section class="mt-10 grid grid-cols-1 gap-6">
	<div class="rounded-lg border border-dashed border-border-primary bg-primary px-4 py-3">
		<Sunburst class="min-h-96 w-full" />
	</div>
</section>

<!-- Sunburst and Scatter Plot (2 columns) -->
<section class="mt-10 grid grid-cols-2 gap-6">
	<div class="rounded-lg border border-dashed border-border-primary bg-primary px-4 py-3">
		<BoxPlot class="min-h-96 w-full" />
	</div>
	
	<div class="rounded-lg border border-dashed border-border-primary bg-primary px-4 py-3">
		<ScatterPlot class="min-h-96 w-full" />
	</div>
</section>

<!-- Box Plot only (full width, 1 column) -->
<section class="mt-10 grid grid-cols-1 gap-6">
	<div class="rounded-lg border border-dashed border-border-primary bg-primary px-4 py-3">
		<TwoHistogram class="min-h-96 w-full" />
	</div>
</section>

</main>
