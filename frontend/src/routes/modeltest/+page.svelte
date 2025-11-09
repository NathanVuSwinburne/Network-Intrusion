<script lang="ts">
	import LineMdUploadOutlineLoop from '../../components/icons/LineMdUploadOutlineLoop.svelte';
	import FileUpload from "../../components/layout/FileUpload.svelte";
	import TextInput from "../../components/ui/TextInput.svelte";
	import {Button} from "bits-ui";
	import PieChart from "../../components/ui/PieChart.svelte";
	import {error} from "@sveltejs/kit";
	import DropdownInput from "../../components/ui/DropdownInput.svelte";

	let loading = $state(false)
	let errors: { [key: string]: string } = $state({})
	let majorError = $state("")
	let showChart = $state(false)
	let chartData: { value: number, name: string}[] = $state([])
	let predictedResult = $state('')

	const ORDER = ['source_bytes', 'dest_bytes', 'source_pkts', 'dest_pkts', 'tcp_win_fwd', 'tcp_win_bwd', 'mean_seg_size_fwd', 'mean_seg_size_bwd', 'duration', 'protocol', 'state']
	const formSubmit = async (event: SubmitEvent) => {
		event.preventDefault()
		const data = new FormData(event.target)
		const input = Object.fromEntries(data.entries())
		let payload: { [key: string]: any } = {}

		// let csvStr = ''

		// Format input separated by commas
		for (let i = 0; i < ORDER.length; i++) {
			const inputKey = ORDER[i]
			const value = input[inputKey] as string

			// Validate each value
			if (value === "") {
				errors[inputKey] = "Please fill out this value."
			} else {
				delete errors[inputKey]
			}

			if (!isNaN(parseFloat(value))) {
				payload[inputKey] = parseFloat(value)
			} else {
				payload[inputKey] = value
			}

			// if (i < ORDER.length -1) {
			// 	csvStr += value + ","
			// } else csvStr += value
		}

		console.log(payload)

		const hasErrors = Object.keys(errors).length > 0
		if (hasErrors) {
			console.log('Failed because of errors')
			return
		} else console.log('Passing request')

		loading = true

		// Get response from backend
		const response = await fetch('http://127.0.0.1:8000/predict/', {
			method: 'POST',
			headers: {
				accept: 'application/json',
				"Content-Type": "application/json"
			},
			body: JSON.stringify(payload)
		});

		const responseData = await response.json()
		if (responseData.error) {
			majorError = responseData.error
			showChart = false
		} else {
			chartData = [
				{ value: Math.floor(responseData.predicted_prob_attack*100), name: 'Malicious' },
				{ value: Math.floor(responseData.predicted_prob_benign*100), name: 'Benign'},
			]
			predictedResult = responseData.predicted_class
			showChart = true
		}

		loading = false
	}


</script>

<main class="container">
	<section class="mt-32 text-center">
		<h1 class="text-3xl font-semibold text-text-primary">Model Test Input</h1>
		<h2 class="text-text-secondary">Provide a sample piece of data and get a response on the status of the network activity</h2>
	</section>

	<section class="flex flex-col items-center justify-center my-32">
		{#if majorError !== ""}
			<div class=" bg-red-200 w-[40rem] px-4 py-2 rounded border-red-500 border mb-8">
				<h3 class="text-red-800 font-semibold">An error occured:</h3>
				<p class="text-red-600 text-sm">{majorError}</p>
			</div>
		{/if}

		<form on:submit={formSubmit} class="mb-26 min-h-[20rem] min-w-[40rem] items-center justify-center rounded-2xl border-2 border-dashed border-border-primary bg-primary p-8 shadow-2xl shadow-[#202020] flex flex-col">
			<h1 class="text-xl font-semibold text-text-primary">Manual Input</h1>
			<div class="grid grid-cols-4 gap-3 my-4">
				<TextInput name="Source Bytes" id="source_bytes" errors={errors} />
				<TextInput name="Destination Bytes" id="dest_bytes" errors={errors} />
				<TextInput name="Source Packets" id="source_pkts" errors={errors} />
				<TextInput name="Destination Packets" id="dest_pkts" errors={errors} />

				<TextInput name="TCP Win Forward" id="tcp_win_fwd" errors={errors} />
				<TextInput name="TCP Win Backward" id="tcp_win_bwd" errors={errors} />
				<TextInput name="Mean Seg Size Forward" id="mean_seg_size_fwd" errors={errors} />
				<TextInput name="Mean Seg Size Backward" id="mean_seg_size_bwd" errors={errors} />

				<TextInput name="Duration(s)" id="duration" errors={errors} />
				<DropdownInput name="Protocol" id="protocol" options={["tcp", "udp", "arp", "ospf", "icmp"]} errors={errors} />
				<DropdownInput name="State" id="state" options={["TXD", "FIN", "CON", "REQ", "INT"]} errors={errors} />
			</div>

			<Button.Root class={`rounded mt-12 px-4 py-2 font-semibold text-primary active:transition-all ${loading ? 'bg-text-secondary' : 'bg-text-primary active:scale-[0.98]'}`} type="submit" disabled={loading}>{loading ? 'Loading...' : 'Submit'}</Button.Root>

	<!--		Center text-->
	<!--		<div class="relative flex h-max w-max flex-col items-center justify-center space-y-3 pb-8 text-text-primary bg-red-400">-->
	<!--			<LineMdUploadOutlineLoop class="h-16 w-16" />-->
	<!--			<div class="text-center">-->
	<!--				<label for="file_upload" class="font text-2xl font-semibold">Upload CSV</label>-->
	<!--				<h3 class="font-light text-text-secondary">Drag or click to upload a file</h3>-->
	<!--			</div>-->
	<!--		</div>-->
		</form>



		{#if showChart}
			<div class="min-h-[20rem] min-w-[40rem] items-center justify-center rounded-2xl border-2 border-dashed border-border-primary bg-primary p-8 shadow-2xl shadow-[#202020] flex flex-col">
				<h1 class="text-xl font-semibold text-text-primary">Results Output</h1>
				<h1 class="font-semibold text-text-primary">Predicted: {predictedResult}</h1>


				<PieChart class="min-h-96 w-full" data={chartData} />
			</div>
		{/if}
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