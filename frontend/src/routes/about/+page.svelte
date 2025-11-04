<script>
    import ScreenIcon from "../../components/icons/ScreenIcon.svelte";
    import CloudIcon from "../../components/icons/CloudIcon.svelte";
    import SheetIcon from "../../components/icons/SheetIcon.svelte";
</script>

<main class="container">
    <section class="my-32 text-center">
        <h1 class="text-3xl font-semibold text-text-primary">About</h1>
        <h2 class="text-text-secondary">Network Anomaly Detection System</h2>
    </section>

    <section class="flex justify-center space-x-5">
        <div class="rounded-lg border border-dashed border-border-primary bg-primary px-4 py-3 max-w-96 h-full flex flex-col items-center">
            <ScreenIcon />
            <h3 class="text-text-primary text-lg font-semibold text-center">Frontend</h3>
            <p class="text-text-secondary text-center">Developed using svelte and tailwindcss</p>
        </div>

        <div class="rounded-lg border border-dashed border-border-primary bg-primary px-4 py-3 max-w-96 h-full flex flex-col items-center">
            <CloudIcon />
            <h3 class="text-text-primary text-lg font-semibold text-center">Backend</h3>
            <p class="text-text-secondary text-center">Implemented in Python with FastAPI</p>
        </div>


        <div class="rounded-lg border border-dashed border-border-primary bg-primary px-4 py-3 max-w-96 h-full flex flex-col items-center">
            <SheetIcon />
            <h3 class="text-text-primary text-lg font-semibold text-center">Model</h3>
            <p class="text-text-secondary text-center">Trained on UNSW-NB15 and CIC-IDS-2017</p>
        </div>
    </section>
</main>