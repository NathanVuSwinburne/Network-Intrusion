<script lang="ts">
    import * as echarts from 'echarts';
    import createRandomString from "$lib/createRandomString";
    import theme from "../../lib/assets/chart-theme.json"

    let { ...others } = $props();

    const chartId = 'pieChart' + createRandomString(4)



    $effect(() => {
        const chartDOM = document.getElementById(chartId);

        echarts.registerTheme("dark", theme)


        const chart = echarts.init(chartDOM, "dark");

        const options =  {
            xAxis: {
                type: 'category',
                data: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            },
            yAxis: {
                type: 'value'
            },
            series: [
                {
                    data: [120, 200, 150, 80, 70, 110, 130],
                    type: 'bar',
                    barCategoryGap: 0
                }
              ],
            grid: { left: 0, right: 0, top: 20, bottom: 25 }
        }

        chart.setOption(options);

        let resizeChart = () => {
            chart.resize();
        };
        window.addEventListener('resize', resizeChart);

        return () => {
            chart.dispose();
            window.removeEventListener('resize', resizeChart);
        };
    });
</script>

<div id={chartId} {...others}></div>