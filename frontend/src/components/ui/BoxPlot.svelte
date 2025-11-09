<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { browser } from '$app/environment';
  import createRandomString from '$lib/createRandomString';
  import theme from '../../lib/assets/chart-theme.json';

  let { data, ...others } = $props();

  type FeatureKey = 'duration' | 'bytes_per_sec' | 'avg_pkt_size' | 'pkts_per_sec';

  const features: FeatureKey[] = ['duration', 'bytes_per_sec', 'avg_pkt_size', 'pkts_per_sec'];
  let selectedFeature: FeatureKey = 'duration';

  const chartId = 'boxPlot' + createRandomString(4);
  let container: HTMLDivElement;        // <-- ref to the DOM node
  let chart: any = null;
  let echartsMod: any = null;

  // Example scaled data (replace with real data)
  // const featureData: Record<FeatureKey, { normal: number[]; anomaly: number[] }> = {
  //   duration: { normal: [0.2,0.25,0.3,0.35,0.4,0.45,0.5], anomaly: [0.5,0.55,0.6,0.65,0.7,0.8,0.9] },
  //   bytes_per_sec: { normal: [0.1,0.12,0.14,0.15,0.16,0.18], anomaly: [0.6,0.65,0.7,0.75,0.8] },
  //   avg_pkt_size: { normal: [0.2,0.25,0.3,0.32,0.35,0.37], anomaly: [0.55,0.6,0.62,0.65,0.7] },
  //   pkts_per_sec: { normal: [0.15,0.18,0.2,0.25,0.28], anomaly: [0.5,0.55,0.6,0.65,0.7] }
  // };

  function getBoxplotData(values: { [key: string]: number}) {
    // const s = [...values].sort((a,b) => a - b);
    // const q1 = s[Math.floor(s.length * 0.25)];
    // const q2 = s[Math.floor(s.length * 0.5)];
    // const q3 = s[Math.floor(s.length * 0.75)];
    // return [s[0], q1, q2, q3, s[s.length - 1]];
    return [values.min, values.q1, values.median, values.q3, values.max]
  }

  function buildOptions() {
    const normalBox = getBoxplotData(data[selectedFeature].BENIGN);
    const anomalyBox = getBoxplotData(data[selectedFeature].ANOMALY);

    return {
      title: {
        text: `Distribution of ${selectedFeature} by Traffic Class`,
        left: 'center',
        textStyle: { color: '#eee', fontSize: 15 }
      },
      tooltip: {
        trigger: 'item',
        formatter: (p: any) => {
          const [id, min, q1, med, q3, max] = p.value;
          return `<b>${p.name}</b><br/>min: ${min}<br/>Q1: ${q1}<br/>median: ${med}<br/>Q3: ${q3}<br/>max: ${max}`;
        }
      },
      xAxis: { type: 'category', data: ['Normal', 'Anomaly'],

        nameTextStyle: {
          color: '#fff',
          fontSize: 14,
        },
        axisLabel: { color: '#fff' },
      },
      yAxis: {
        type: 'value',
        name: 'Scaled Value',
        nameLocation: 'middle',
        nameGap: 35,
        axisLabel: { color: '#fff ' },
        splitLine: { show: true },
        nameTextStyle: {
          color: '#fff',
          fontSize: 14,
        }
      },
      colorBy: 'series',                 // don't let palette override per-item color
      series: [
      {
        name: 'Traffic Class',
        type: 'boxplot',
        boxWidth: [20, 60],
        // keep palette from overriding per-item color
        colorBy: 'series',
        data: [
          {
            name: 'Normal',
            value: normalBox,
            itemStyle: {
              color: 'rgba(33,150,243,0.35)',   // blue fill, slightly transparent
              borderColor: '#2196F3',           // blue border = whiskers color
              borderWidth: 2                    // <-- makes whisker/min/max lines always visible
            },
            emphasis: { disabled: true }
          },
          {
            name: 'Anomaly',
            value: anomalyBox,
            itemStyle: {
              color: 'rgba(229,57,53,0.35)',    // red fill, slightly transparent
              borderColor: '#E53935',           // red border = whiskers color
              borderWidth: 2
            },
            emphasis: { disabled: true }
          }
        ]
      }
    ],
      grid: { left: 60, right: 40, top: 60, bottom: 50 }
    };
  }

  async function ensureChart() {
    if (!browser || !container) return;
    if (!echartsMod) {
      echartsMod = (await import('echarts'));
      echartsMod.registerTheme('dark', theme);
    }
    if (!chart) chart = echartsMod.init(container, 'dark');
  }

  async function renderChart() {
    await ensureChart();
    if (!chart) return;
    chart.setOption(buildOptions());
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

  // re-render when dropdown changes
  $effect(() => {
    // touch the dependency so the effect tracks it
    void selectedFeature;
    renderChart();
  });
</script>

<!-- Controls -->
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
<div id={chartId} bind:this={container} {...others}></div>
