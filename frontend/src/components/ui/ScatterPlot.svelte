<script lang="ts">
	import * as echarts from "echarts";
	import createRandomString from "$lib/createRandomString";
	import theme from "../../lib/assets/chart-theme.json";

	let { ...others } = $props();

	const chartId = "scatterChart" + createRandomString(4);

	$effect(() => {
		const chartDOM = document.getElementById(chartId);
		echarts.registerTheme("dark", theme);
		const chart = echarts.init(chartDOM, "dark");

		// --- Example data: two clusters (normal vs anomaly) ---
		// You can later replace this with dynamic data from API or props
		const normalData = [
			[50, 300],
			[70, 400],
			[60, 420],
			[80, 350],
			[90, 500],
			[100, 550],
		];

		const anomalyData = [
			[200, 1500],
			[230, 1600],
			[250, 1700],
			[270, 1800],
			[290, 1900],
			[310, 2100],
		];

		const options = {
			title: {
				text: "Packet Size vs Bytes/sec",
				left: "center",
				textStyle: {
					color: "#eee",
					fontSize: 14,
				},
			},
			tooltip: {
				trigger: "item",
				formatter: (params: any) => {
					const [x, y] = params.value;
					return `
						<b>${params.seriesName}</b><br/>
						avg_pkt_size: ${x}<br/>
						bytes_per_sec: ${y}
					`;
				},
			},
			legend: {
				top: 30,
				data: ["Normal", "Anomaly"],
				textStyle: {
					color: "#ccc",
				},
			},
			xAxis: {
				name: "avg_pkt_size",
				nameLocation: "middle",
				nameGap: 30,
				type: "value",
				splitLine: { show: true },
			},
			yAxis: {
				name: "bytes_per_sec",
				nameLocation: "middle",
				nameGap: 45,
				type: "value",
				splitLine: { show: true },
			},
			series: [
				{
					name: "Normal",
					type: "scatter",
					data: normalData,
					symbolSize: 10,
					itemStyle: {
						color: "#4CAF50", // green
					},
				},
				{
					name: "Anomaly",
					type: "scatter",
					data: anomalyData,
					symbolSize: 10,
					itemStyle: {
						color: "#E53935", // red
					},
				},
			],
		};

		chart.setOption(options);

		let resizeChart = () => chart.resize();
		window.addEventListener("resize", resizeChart);

		return () => {
			chart.dispose();
			window.removeEventListener("resize", resizeChart);
		};
	});
</script>

<div id={chartId} {...others}></div>
