<script lang="ts">
  import { onMount } from 'svelte';
  import { browser } from '$app/environment';
  import createRandomString from '$lib/createRandomString';
  import theme from '../../lib/assets/chart-theme.json';

  let { ...others } = $props();

  type FeatureKey = 'avg_pkt_size' | 'duration' | 'bytes_per_sec' | 'pkts_per_sec';

  const features: FeatureKey[] = ['avg_pkt_size', 'duration', 'bytes_per_sec', 'pkts_per_sec'];
  let selectedFeature: FeatureKey = 'avg_pkt_size';

  const chartId = 'hist_' + createRandomString(4);
  let container: HTMLDivElement;
  let chart: any = null;
  let echartsMod: any = null;

  // Example data (replace with your real arrays; values can be scaled already)
  const featureData: Record<FeatureKey, { normal: number[]; anomaly: number[] }> = {
    avg_pkt_size: {
      normal: [2,4,5,6,6.4,6.8,7,7.2,8.5,9.5,11,12.8],
      anomaly: [1.5,3,4,5.5,6.2,7.8,8,8.2,9,10.5,11.2,13.5]
    },
    duration: {
      normal: [0.2,0.25,0.3,0.35,0.4,0.45,0.5],
      anomaly: [0.5,0.55,0.6,0.65,0.7,0.8,0.9]
    },
    bytes_per_sec: {
      normal: [0.1,0.12,0.14,0.15,0.16,0.18],
      anomaly: [0.6,0.65,0.7,0.75,0.8]
    },
    pkts_per_sec: {
      normal: [0.15,0.18,0.2,0.25,0.28],
      anomaly: [0.5,0.55,0.6,0.65,0.7]
    }
  };

  function makeHistogram(values: number[], binCount = 7) {
    if (values.length === 0) return { labels: [], counts: [] };

    const min = Math.min(...values);
    const max = Math.max(...values);
    const width = (max - min) || 1; // avoid zero width
    const step = width / binCount;

    const edges = Array.from({ length: binCount + 1 }, (_, i) => min + i * step);
    const counts = Array(binCount).fill(0);

    for (const v of values) {
      let idx = Math.floor((v - min) / step);
      if (idx >= binCount) idx = binCount - 1; // include max in last bin
      if (idx < 0) idx = 0;
      counts[idx]++;
    }

    const labels = Array.from({ length: binCount }, (_, i) => {
      const a = edges[i];
      const b = edges[i + 1];
      // nicer tick labels (short)
      return `${Number(a.toFixed(2))}–${Number(b.toFixed(2))}`;
    });

    return { labels, counts };
  }

  function buildOptions() {
    const { normal, anomaly } = featureData[selectedFeature];

    const bins = 7; // tweak if desired
    const normalHist = makeHistogram(normal, bins);
    const anomalyHist = makeHistogram(anomaly, bins);

    return {
      // main title
      title: [
        {
          text: `Distribution of ${selectedFeature} by Traffic Class`,
          left: 'center',
          top: 10,
          textStyle: { color: '#eee', fontSize: 16, fontWeight: 'bold' }
        },
        // sub-titles above each small chart
        { text: 'Normal Class', left: '25%', top: 50, textStyle: { color: '#ddd', fontSize: 14 } },
        { text: 'Anomaly Class', left: '75%', top: 50, textStyle: { color: '#ddd', fontSize: 14 } }
      ],
      tooltip: { trigger: 'axis' },
      // two grids side-by-side
      grid: [
        { left: 50, right: '55%', top: 80, bottom: 60 },
        { left: '55%', right: 50, top: 80, bottom: 60 }
      ],
      xAxis: [
        {
          type: 'category',
          gridIndex: 0,
          data: normalHist.labels,
          name: selectedFeature,
          nameLocation: 'middle',
          nameGap: 30,
          axisLabel: { color: '#ccc', interval: 0, rotate: 0 }
        },
        {
          type: 'category',
          gridIndex: 1,
          data: anomalyHist.labels,
          name: selectedFeature,
          nameLocation: 'middle',
          nameGap: 30,
          axisLabel: { color: '#ccc', interval: 0, rotate: 0 }
        }
      ],
      yAxis: [
        {
          type: 'value',
          gridIndex: 0,
          name: 'frequency',
          nameLocation: 'middle',
          nameGap: 40,
          axisLabel: { color: '#ccc' },
          splitLine: { show: true }
        },
        {
          type: 'value',
          gridIndex: 1,
          name: 'frequency',
          nameLocation: 'middle',
          nameGap: 40,
          axisLabel: { color: '#ccc' },
          splitLine: { show: true }
        }
      ],
      // keep per-series color stable
      colorBy: 'series',
      series: [
        {
          name: 'Normal',
          type: 'bar',
          xAxisIndex: 0,
          yAxisIndex: 0,
          data: normalHist.counts,
          itemStyle: { color: '#1E40AF' }, // blue
          barCategoryGap: '20%',
          barWidth: '60%'
        },
        {
          name: 'Anomaly',
          type: 'bar',
          xAxisIndex: 1,
          yAxisIndex: 1,
          data: anomalyHist.counts,
          itemStyle: { color: '#E53935' }, // red
          barCategoryGap: '20%',
          barWidth: '60%'
        }
      ]
    };
  }

  async function ensureChart() {
    if (!browser || !container) return;
    if (!echartsMod) {
      echartsMod = await import('echarts');
      echartsMod.registerTheme('dark', theme);
    }
    if (!chart) chart = echartsMod.init(container, 'dark');
  }

  async function renderChart() {
    await ensureChart();
    if (!chart) return;
    chart.setOption(buildOptions(), true);
    chart.resize();
  }

  onMount(() => {
    renderChart();
    const onResize = () => chart?.resize();
    window.addEventListener('resize', onResize);
    return () => {
      window.removeEventListener('resize', onResize);
      chart?.dispose();
      chart = null;
    };
  });

  // Re-render when feature changes (Svelte 5 runes)
  $effect(() => {
    void selectedFeature; // track dependency
    renderChart();
  });
</script>

<!-- Controls -->
<div class="flex items-center gap-3 mb-3">
  <label class="text-text-primary font-medium">Choose feature</label>
  <select
    bind:value={selectedFeature}
    on:change={renderChart}
    class="rounded-md bg-primary border border-border-primary px-2 py-1 text-text-primary"
  >
    {#each features as f}<option value={f}>{f}</option>{/each}
  </select>
</div>

<!-- One canvas with two sub-charts -->
<div id={chartId} bind:this={container} {...others}></div>
