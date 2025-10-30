<script>
	import IconesCSVFile from '../../components/icons/IconesCSVFile.svelte';
	import PieChart from "../../components/ui/PieChart.svelte";
	import Histogram from "../../components/ui/Histogram.svelte";
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

	<section class="mt-10 grid grid-cols-2 gap-6">
		<div class="rounded-lg border border-dashed border-border-primary bg-primary px-4 py-3 ">
			<h3 class="text-text-primary">Class Distribution</h3>

			<PieChart class="min-h-96" />
		</div>


		<div class="rounded-lg border border-dashed border-border-primary bg-primary px-4 py-3">
			<h3 class="text-text-primary">Class Distribution</h3>

			<Histogram class="min-h-96" />
		</div>

		<div class="rounded-lg border border-dashed border-border-primary bg-primary px-4 py-3">
			<h3 class="text-text-primary">Class Distribution</h3>


		</div>
	</section>
</main>
