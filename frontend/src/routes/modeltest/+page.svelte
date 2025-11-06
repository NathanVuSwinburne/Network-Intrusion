<script lang="ts">
	import LineMdUploadOutlineLoop from '../../components/icons/LineMdUploadOutlineLoop.svelte';
	import FileUpload from "../../components/layout/FileUpload.svelte";
	import TextInput from "../../components/ui/TextInput.svelte";
	import {Button} from "bits-ui";
	import PieChart from "../../components/ui/PieChart.svelte";

	const ORDER = ['src_bytes', 'dest_bytes', 'src_pkts', 'dest_pkts', 'tcp_win_fwd', 'tcp_win_bwd', 'seg_size_fwd', 'seg_size_bwd', 'dur', 'proto_enc', 'state_enc']
	const formSubmit = async (event: SubmitEvent) => {
		event.preventDefault()
		const data = new FormData(event.target)
		const input = Object.fromEntries(data.entries())

		let csvStr = ''

		// Format input separated by commas
		for (let i = 0; i < ORDER.length; i++) {
			const inputKey = ORDER[i]
			if (i < ORDER.length -1) {
				csvStr += input[inputKey] + ","
			} else csvStr += input[inputKey]
		}


		// Get response from backend
		const response = await fetch('http://127.0.0.1:8000', {
			method: 'POST',
			body: csvStr
		});

		console.log(response)
	}


</script>

<main class="container">
	<section class="mt-32 text-center">
		<h1 class="text-3xl font-semibold text-text-primary">Model Test Input</h1>
		<h2 class="text-text-secondary">Provide a sample piece of data and get a response on the status of the network activity</h2>
	</section>

	<section class="flex flex-col space-y-26 items-center justify-center my-32">

		<form on:submit={formSubmit} class="min-h-[20rem] min-w-[40rem] items-center justify-center rounded-2xl border-2 border-dashed border-border-primary bg-primary p-8 shadow-2xl shadow-[#202020] flex flex-col">
			<h1 class="text-xl font-semibold text-text-primary">Manual Input</h1>
			<div class="grid grid-cols-4 gap-3 my-4">
				<TextInput name="Source Bytes" id="src_bytes" />
				<TextInput name="Destination Bytes" id="dest_bytes" />
				<TextInput name="Source Packets" id="src_pkts" />
				<TextInput name="Destination Packets" id="dest_pkts" />

				<TextInput name="TCP Win Forward" id="tcp_win_fwd" />
				<TextInput name="TCP Win Backward" id="tcp_win_bwd" />
				<TextInput name="Mean Seg Size Forward" id="seg_size_fwd" />
				<TextInput name="Mean Seg Size Backward" id="seg_size_bwd" />

				<TextInput name="Duration" id="dur" />
				<TextInput name="Protocol Encoded" id="proto_enc" />
				<TextInput name="State Encoded" id="state_enc" />
			</div>

			<Button.Root class="rounded mt-12 bg-text-primary px-4 py-2 font-semibold text-primary active:transition-all active:scale-[0.98]" type="submit">Submit</Button.Root>

	<!--		Center text-->
	<!--		<div class="relative flex h-max w-max flex-col items-center justify-center space-y-3 pb-8 text-text-primary bg-red-400">-->
	<!--			<LineMdUploadOutlineLoop class="h-16 w-16" />-->
	<!--			<div class="text-center">-->
	<!--				<label for="file_upload" class="font text-2xl font-semibold">Upload CSV</label>-->
	<!--				<h3 class="font-light text-text-secondary">Drag or click to upload a file</h3>-->
	<!--			</div>-->
	<!--		</div>-->
		</form>



		<div class="min-h-[20rem] min-w-[40rem] items-center justify-center rounded-2xl border-2 border-dashed border-border-primary bg-primary p-8 shadow-2xl shadow-[#202020] flex flex-col">
			<h1 class="text-xl font-semibold text-text-primary">Results Output</h1>

			<PieChart class="min-h-96 w-full" />
		</div>
	</section>

<!--	<section id="dataset-features" class="container flex items-center">-->
<!--		<div class="w-full">-->
<!--			<h2 class="text-3xl font-semibold text-text-primary text-center">Dataset Features</h2>-->
<!--			<p class="text-text-secondary text-center">These features are required in order for our model to process your data</p>-->
<!--		</div>-->

<!--		<table>-->
<!--			<thead>-->
<!--			<tr>-->
<!--				<th>avg_pkt_size</th>-->
<!--				<th>mean_seg_size_bwd</th>-->
<!--				<th>dest_bytes</th>-->
<!--				<th>source_pkts</th>-->
<!--				<th>dest_pkts</th>-->
<!--				<th>mean_seg_size_fwd</th>-->
<!--				<th>mean_seg_i</th>-->
<!--			</tr>-->
<!--			</thead>-->
<!--		</table>-->
<!--	</section>-->
</main>

<style>
	::file-selector-button {
		display: none;
	}
</style>