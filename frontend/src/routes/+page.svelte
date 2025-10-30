<script>
	import DottedArrowDown from '../components/icons/DottedArrowDown.svelte';
	import FeatureCard from "../components/ui/FeatureCard.svelte";
	import {Button} from "bits-ui";

	const HEADING = "Network Anomaly Detection"
</script>

<main class="container pt-8">
	<div class="flex h-[90vh] flex-col items-center justify-center">
		<h1 class="text-4xl flex font-semibold text-text-primary">
			{#each HEADING as letter}
				{#if letter === ' '}
					<div class="w-5"></div>
				{:else}
					<span class="inline-block hover:-translate-y-1 transition-all">{letter}</span>
				{/if}
			{/each}
		</h1>
		<p class="max-w-[40rem] text-center text-wrap text-text-secondary">
			A model trained on vast datasets of network behavior, it identifies subtle anomalies that
			traditional systems often miss.
		</p>
		<div class="mt-6 flex justify-center items-center gap-6">
			<Button.Root class="rounded bg-text-primary px-4 py-2 font-semibold text-primary active:transition-all active:scale-[0.98]" href="upload">Get Started</Button.Root>
			<Button.Root class="rounded border-dashed px-4 py-2 text-text-primary border border-border-primary active:transition-all active:scale-[0.98]" href="#features">Features</Button.Root>
		</div>

<!--		<div class="absolute bottom-44 flex flex-col items-center">-->
<!--			<p class="text-sm font-medium text-text-secondary">Scroll to view more</p>-->
<!--			<DottedArrowDown class="mt-2 h-[2rem] w-[2rem] animate-bounce text-text-secondary" />-->
<!--		</div>-->
	</div>
	<section id="features" class="flex flex-col items-center space-y-4">
		<h2 class="text-3xl font-semibold text-text-primary">Features</h2>
		<div class="grid grid-cols-3 gap-5">
			<FeatureCard title="Feature 1" desc="Test test test test test test test test test" />
			<FeatureCard title="Feature 1" desc="Test test test test test" />
			<FeatureCard title="Feature 1" desc="Test test test test test test test test test" />
		</div>
	</section>
	<div class="h-[40vh]"></div>
</main>
