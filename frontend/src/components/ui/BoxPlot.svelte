<script lang="ts">
	import * as echarts from "echarts";
	import createRandomString from "$lib/createRandomString";
	import theme from "../../lib/assets/chart-theme.json";

    let { ...others } = $props();

    type FeatureKey = "duration" | "bytes_per_sec" | "avg_pkt_size" | "pkts_per_sec";

	// Dropdown options
	const features: FeatureKey[] = [
		"duration",
		"bytes_per_sec",
		"avg_pkt_size",
		"pkts_per_sec"
	];

	// selected feature
	let selectedFeature: FeatureKey = "duration";
	const chartId = "boxPlot" + createRandomString(4);

	let chart: echarts.ECharts | null = null;

	// Dummy scaled data for example
	// Replace with your own computed data later
	const featureData: Record<FeatureKey, { normal: number[]; anomaly: number[] }> = {
		duration: {
			normal: [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5],
			anomaly: [0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.9]
		},
		bytes_per_sec: {
			normal: [0.1, 0.12, 0.14, 0.15, 0.16, 0.18],
			anomaly: [0.6, 0.65, 0.7, 0.75, 0.8]
		},
		avg_pkt_size: {
			normal: [0.2, 0.25, 0.3, 0.32, 0.35, 0.37],
			anomaly: [0.55, 0.6, 0.62, 0.65, 0.7]
		},
		pkts_per_sec: {
			normal: [0.15, 0.18, 0.2, 0.25, 0.28],
			anomaly: [0.5, 0.55, 0.6, 0.65, 0.7]
		}
	};

	// Function to compute ECharts boxplot input
	function getBoxplotData(values: number[]) {
		const sorted = [...values].sort((a, b) => a - b);
		const q1 = sorted[Math.floor(sorted.length * 0.25)];
		const q2 = sorted[Math.floor(sorted.length * 0.5)];
		const q3 = sorted[Math.floor(sorted.length * 0.75)];
		const min = sorted[0];
		const max = sorted[sorted.length - 1];
		return [min, q1, q2, q3, max];
	}

	function renderChart() {
		const chartDOM = document.getElementById(chartId);
		if (!chartDOM) return;

		if (!chart) {
			echarts.registerTheme("dark", theme);
			chart = echarts.init(chartDOM, "dark");
		}

		const normalBox = getBoxplotData(featureData[selectedFeature].normal);
		const anomalyBox = getBoxplotData(featureData[selectedFeature].anomaly);

		const options = {
			title: {
				text: `Distribution of ${selectedFeature} by Traffic Class`,
				left: "center",
				textStyle: { color: "#eee", fontSize: 15 }
			},
			tooltip: { trigger: "item" },
			xAxis: {
				type: "category",
				data: ["Normal", "Anomaly"],
				axisLabel: { color: "#ccc" }
			},
			yAxis: {
				type: "value",
				name: "Scaled Value",
				nameLocation: "middle",
				nameGap: 35,
				axisLabel: { color: "#ccc" },
				splitLine: { show: true }
			},
			series: [
				{
					name: "Traffic Class",
					type: "boxplot",
					data: [normalBox, anomalyBox],
					itemStyle: {
						color: (params: any) =>
							params.dataIndex === 0 ? "#2196F3" : "#E53935" // blue/red
					},
					boxWidth: [20, 60]
				}
			],
			grid: { left: 60, right: 40, top: 60, bottom: 50 }
		};

		chart.setOption(options);
		chart.resize();
	}

	// Initialize and rerender when feature changes
	$effect(() => {
		renderChart();
	});

	window.addEventListener("resize", () => chart?.resize());
</script>

<!-- Dropdown selector -->
<div class="flex items-center justify-between mb-4">
	<label class="text-text-primary font-medium"></label>
	<select
		bind:value={selectedFeature}
		on:change={renderChart}
		class="rounded-md bg-primary border border-border-primary px-2 py-1 text-text-primary"
	>
		{#each features as feature}
			<option value={feature}>{feature}</option>
		{/each}
	</select>
</div>

<!-- Chart container -->
<div id={chartId} {...others}></div>
