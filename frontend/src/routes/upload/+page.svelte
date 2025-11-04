<script>
	import LineMdUploadOutlineLoop from '../../components/icons/LineMdUploadOutlineLoop.svelte';
	import FileUpload from "../../components/layout/FileUpload.svelte";

</script>

<section class="flex h-screen items-center justify-center">
		<FileUpload />
</section>
<section id="dataset-features" class="container flex items-center">
	<div class="w-full">
		<h2 class="text-3xl font-semibold text-text-primary text-center">Dataset Features</h2>
		<p class="text-text-secondary text-center">These features are required in order for our model to process your data</p>
	</div>

	<table>
		<thead>
		<tr>
			<th>avg_pkt_size</th>
			<th>mean_seg_size_bwd</th>
			<th>dest_bytes</th>
			<th>source_pkts</th>
			<th>dest_pkts</th>
			<th>mean_seg_size_fwd</th>
			<th>mean_seg_i</th>
		</tr>
		</thead>
	</table>
</section>

<style>
	::file-selector-button {
		display: none;
	}
</style>
