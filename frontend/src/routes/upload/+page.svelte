<script>
	import LineMdUploadOutlineLoop from '../../components/icons/LineMdUploadOutlineLoop.svelte';
	import FileUpload from "../../components/layout/FileUpload.svelte";

	let manual = $state(false);
</script>

<main class="flex h-screen w-screen items-center justify-center">
	{#if manual}
		<FileUpload />
	{:else}

	{/if}

</main>

<style>
	::file-selector-button {
		display: none;
	}
</style>
