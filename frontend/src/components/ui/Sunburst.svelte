<script lang="ts">
  import { onMount } from 'svelte';
  import { browser } from '$app/environment';
  import createRandomString from '$lib/createRandomString';
  import theme from '../../lib/assets/chart-theme.json';

  let { data, ...others } = $props();

  const chartId = 'sunburst_' + createRandomString(4);
  let container: HTMLDivElement;
  let chart: any = null;
  let echartsMod: any = null;

  // --- Replace with your real counts ---
  // Provide 4 states per class.
  // Values are counts (not percentages) — the chart computes percentages.
  // const breakdown = {
  //   normal: { stateA: 16, stateB: 12, stateC: 9, stateD: 5 },
  //   anomaly: { stateA: 8,  stateB: 6,  stateC: 4, stateD: 3 }
  // };

  const stateLabels = {
    stateA: 'State A',
    stateB: 'State B',
    stateC: 'State C',
    stateD: 'State D'
  };

  // Colors
  const COLOR_NORMAL = '#1E40AF'; // blue
  const COLOR_ANOMALY = '#E53935'; // red

  function sumValues(obj: Record<string, number>) {
    return Object.values(obj).reduce((a, b) => a + b, 0);
  }

  function buildData() {
    const totalNormal  = sumValues(data.BENIGN);
    const totalAnomaly = sumValues(data.ANOMALY);

    const normalNode = {
      name: 'Normal',
      value: totalNormal,
      itemStyle: { color: COLOR_NORMAL },
      children: Object.entries(data.BENIGN).map(([k, v]) => ({
        name: stateLabels[k as keyof typeof stateLabels] ?? k,
        value: v
      }))
    };

    const anomalyNode = {
      name: 'Anomaly',
      value: totalAnomaly,
      itemStyle: { color: COLOR_ANOMALY },
      children: Object.entries(data.ANOMALY).map(([k, v]) => ({
        name: stateLabels[k as keyof typeof stateLabels] ?? k,
        value: v
      }))
    };

    return {
      data: [normalNode, anomalyNode],
      grandTotal: totalNormal + totalAnomaly
    };
  }

  function buildOptions() {
    const { data, grandTotal } = buildData();

    return {
      title: {
        text: 'Anomaly vs Normal — Sunburst Breakdown',
        left: 'center',
        top: 10,
        textStyle: { color: '#eee', fontSize: 16, fontWeight: 'bold' }
      },
      legend: {
        top: '6%',
        left: 'center',
        selectedMode: false,
        textStyle: { color: '#ddd' },
        data: ['Normal', 'Anomaly']
      },
      tooltip: {
        trigger: 'item',
        formatter: (p: any) => {
          const path = p.treePathInfo; // [root?, class, child]
          const val = p.value as number;

          // % of parent for second layer; % of total for first layer
          let pct: number;
          if (path.length >= 3) {
            const parent = path[path.length - 2];
            pct = parent.value ? (val / parent.value) * 100 : 0;
          } else {
            pct = grandTotal ? (val / grandTotal) * 100 : 0;
          }

          return `<b>${p.name}</b><br/>Count: ${val}<br/>Percent: ${pct.toFixed(1)}%`;
        }
      },
      series: [
        {
          type: 'sunburst',
          radius: ['20%', '85%'],
          sort: 'none',
          emphasis: { focus: 'ancestor' },
          label: {
            rotate: 'radial',
            color: '#eee'
          },
          levels: [
            {}, // placeholder for root
            {
              // Inner ring (Normal / Anomaly)
              r0: '20%',
              r: '50%',
              label: { rotate: 0, fontWeight: 'bold' },
              itemStyle: { borderWidth: 2, borderColor: '#141414' }
            },
            {
              // Outer ring (states)
              r0: '50%',
              r: '85%',
              label: { rotate: 'tangential' },
              itemStyle: { borderWidth: 2, borderColor: '#141414' }
            }
          ],
          data
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
</script>

<div id={chartId} bind:this={container} {...others}></div>
