<script lang="ts">
	import '../app.css';
	import '../fonts.css';
	import favicon from '$lib/assets/favicon.svg';
	import { page } from '$app/state';

	let { children } = $props();

	const ROUTES: { id: string; name: string }[] = [
		{ id: '/', name: 'Home' },
		{ id: '/about', name: 'About' },
		{ id: '/upload', name: 'Upload'}
	];
</script>

<svelte:head>
	<link rel="icon" href={favicon} />
	<title>Network Anomaly Detection</title>
	<meta name="description" content="Network anomaly and packet analysis tool" />
</svelte:head>

<!--<section class="container text-white">-->
<!--	<div class=" w-max rounded-full bg-red-300">-->
<!--		<h1>what</h1>-->
<!--	</div>-->
<!--</section>-->

<section class="fixed top-[2%] left-[50%] container mt-2 translate-[-50%]">
	<nav
		class="flex justify-between rounded-full border border-dashed border-border-primary bg-primary px-4 py-2"
	>
		<h1 class="font-medium text-white">Network Anomaly Detector</h1>
		<div class="flex space-x-5">
			{#each ROUTES as route}
				<a
					href={route.id}
					class={`${
						page.route.id === route.id ? 'text-text-primary' : 'text-text-secondary'
					} border-b-white transition-all hover:border-b hover:text-text-primary`}
				>
					{route.name}
				</a>
			{/each}
		</div>
	</nav>
</section>

{@render children?.()}

<style>
</style>
