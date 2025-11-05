<script>
	import IconesCSVFile from '../../components/icons/IconesCSVFile.svelte';
	import PieChart from "../../components/ui/PieChart.svelte";
	import Histogram from "../../components/ui/Histogram.svelte";
	import TwoHistogram from "../../components/ui/TwoHistogram.svelte";
	import ScatterPlot from '../../components/ui/ScatterPlot.svelte';
	import BoxPlot from '../../components/ui/BoxPlot.svelte';
	import Sunburst from '../../components/ui/Sunburst.svelte';
</script>

<main class="container">
	<section class="mt-32">
		<h1 class="text-3xl font-semibold text-text-primary">Model Results</h1>
		<h2 class="text-text-secondary">Completed at 4:10pm | Took 60s</h2>
	</section>

	<section class="mt-10 grid grid-cols-3 gap-16">
		<div class="rounded-lg border border-dashed border-border-primary bg-primary px-4 py-3 space-y-3">
			<h3 class="text-text-primary">File Statistics</h3>

			<div class="grid grid-cols-2 gap-3">
				<div>
					<h3 class="text-text-primary text-sm">File</h3>
					<p class="text-text-secondary">results_mk1.csv</p>
				</div>

				<div>
					<h3 class="text-text-primary text-sm">Size</h3>
					<p class="text-text-secondary">382mb</p>
				</div>


				<div>
					<h3 class="text-text-primary text-sm">Datapoints</h3>
					<p class="text-text-secondary">1,830,900</p>
				</div>
			</div>
		</div>


		<div class="rounded-lg border border-dashed border-border-primary bg-primary px-4 py-3 space-y-3">
			<h3 class="text-text-primary">File Statistics</h3>

			<div class="grid grid-cols-2 gap-3">
				<div>
					<h3 class="text-text-primary text-sm">File</h3>
					<p class="text-text-secondary">results_mk1.csv</p>
				</div>

				<div>
					<h3 class="text-text-primary text-sm">Size</h3>
					<p class="text-text-secondary">382mb</p>
				</div>


				<div>
					<h3 class="text-text-primary text-sm">Datapoints</h3>
					<p class="text-text-secondary">1,830,900</p>
				</div>
			</div>
		</div>

		<div class="rounded-lg border border-dashed border-border-primary bg-primary px-4 py-3 space-y-3">
			<h3 class="text-text-primary">Result Statistics</h3>

			<div class="grid grid-cols-2 gap-3">
				<div>
					<h3 class="text-text-primary text-sm">Benign Count</h3>
					<p class="text-text-secondary">3,000</p>
				</div>

				<div>
					<h3 class="text-text-primary text-sm">Anomaly Count</h3>
					<p class="text-text-secondary">300</p>
				</div>

			</div>
		</div>
	</section>

	<!-- Histogram only (full width, 1 column) -->
	<section class="mt-10 grid grid-cols-1 gap-6">
	<div class="rounded-lg border border-dashed border-border-primary bg-primary px-4 py-3">
		<h3 class="text-text-primary">Feature Distribution</h3>
		<TwoHistogram class="min-h-96 w-full" />
	</div>
	</section>

	<!-- Sunburst and Scatter Plot (2 columns) -->
	<section class="mt-10 grid grid-cols-2 gap-6">
	<div class="rounded-lg border border-dashed border-border-primary bg-primary px-4 py-3">
		<h3 class="text-text-primary">Class Percentage (Anomaly vs Normal)</h3>
		<Sunburst class="min-h-96 w-full" />
	</div>

	<div class="rounded-lg border border-dashed border-border-primary bg-primary px-4 py-3">
		<h3 class="text-text-primary">Feature Relationship (Scatter Plot)</h3>
		<ScatterPlot class="min-h-96 w-full" />
	</div>
	</section>

	<!-- Box Plot only (full width, 1 column) -->
	<section class="mt-10 grid grid-cols-1 gap-6">
	<div class="rounded-lg border border-dashed border-border-primary bg-primary px-4 py-3">
		<h3 class="text-text-primary">Feature Distribution by Traffic Class</h3>
		<BoxPlot class="min-h-96 w-full" />
	</div>
</section>

</main>
