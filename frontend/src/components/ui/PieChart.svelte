<script lang="ts">
    import * as echarts from 'echarts';
    import createRandomString from "$lib/createRandomString";
    import theme from "../../lib/assets/chart-theme.json"

    let { data, ...others } = $props();

    const chartId = 'pieChart' + createRandomString(4)



    $effect(() => {
        const chartDOM = document.getElementById(chartId);

        echarts.registerTheme("dark", theme)


        const chart = echarts.init(chartDOM, "dark");

        const options =  {
            tooltip: {
                trigger: 'item'
            },
            legend: {
                top: '5%',
                left: 'center'
            },
            series: [
                {
                    name: 'Status Type',
                    type: 'pie',
                    radius: ['40%', '70%'],
                    avoidLabelOverlap: false,
                    padAngle: 5,
                    itemStyle: {
                        borderRadius: 10
                    },
                    label: {
                        show: false,
                        position: 'center'
                    },
                    emphasis: {
                        label: {
                            show: true,
                            fontSize: 60,
                            fontWeight: 'bold',
                            fontFamily: '0xProto Nerd Font Mono'
                        }
                    },
                    labelLine: {
                        show: false
                    },
                    data: data
                }

            ],

        }


        // [
        // { value: 1048, name: 'Search Engine' },
        //     { value: 735, name: 'Direct' },
        // ]                }
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