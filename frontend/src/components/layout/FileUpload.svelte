<script>
import LineMdUploadOutlineLoop from "../icons/LineMdUploadOutlineLoop.svelte";
</script>

<div
        class="absolute flex h-[20rem] w-[40rem] items-center justify-center rounded-2xl border-2 border-dashed border-border-primary bg-primary p-8 shadow-2xl shadow-[#202020]"
>
    <input
            class="absolute top-0 left-0 z-10 h-full w-full bg-green-300 opacity-0"
            type="file"
            id="file_upload"
            name="file_upload"
    />

    <div
            class="relative flex h-max w-max flex-col items-center justify-center space-y-3 pb-8 text-text-primary"
    >
        <LineMdUploadOutlineLoop class="h-16 w-16" />
        <div class="text-center">
            <label for="file_upload" class="font text-2xl font-semibold">Upload CSV</label>
            <h3 class="font-light text-text-secondary">Drag or click to upload a file</h3>
        </div>
    </div>
</div>