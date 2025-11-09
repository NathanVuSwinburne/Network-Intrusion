<script lang="ts">
import LineMdUploadOutlineLoop from "../icons/LineMdUploadOutlineLoop.svelte";
import LoadingIcon from "../icons/LoadingIcon.svelte";

let selectedFile = null;
let result = $state();
let hasError = $state(false)
let loading = $state(false)

const handleFileChange = (event: any) => {
    selectedFile = event.target.files[0];
    if (selectedFile) {
        uploadFile(selectedFile);
    }
};

const uploadFile = async (file: File) => {
    loading = true
    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('http://localhost:8000/upload-csv/', {
            method: 'POST',
            body: formData,
        });

        const jsonResponse = await response.json();
        // result = JSON.parse(jsonResponse, null, 2); // Show the response data
        result = jsonResponse
        if (response.ok) {
            hasError = false
        } else {
            // result = response.statusText;
            hasError = true
        }

    } catch (error) {
        result = error.message;
        hasError = true
    } finally {
        loading = false
    }
};
</script>

<div class="flex flex-col space-y-2">
    {#if hasError}
        <div class=" bg-red-200 w-[40rem] px-4 py-2 rounded border-red-500 border">
            <h3 class="text-red-800 font-semibold">An error occured:</h3>
            <p class="text-red-600 text-sm">{result.message}</p>
        </div>
    {/if}

    <div
            class="flex h-[20rem] w-[40rem] items-center justify-center rounded-2xl border-2 border-dashed border-border-primary bg-primary p-8 shadow-2xl shadow-[#202020]"
    >
        <input
                class="absolute top-0 left-0 z-10 h-full w-full bg-green-300 opacity-0"
                type="file"
                id="file_upload"
                name="file_upload"
                disabled={loading}
                on:change={handleFileChange}
        />

        <div
                class="relative flex h-max w-max flex-col items-center justify-center space-y-3 pb-8 text-text-primary"
        >
            {#if loading}
                <LoadingIcon class="h-16 w-16" />
                <div class="text-center">
                    <label for="file_upload" class="font text-2xl font-semibold">Processing data...</label>
                    <h3 class="font-light text-text-secondary">Please wait while your upload is processed...</h3>
                </div>
            {:else}
                <LineMdUploadOutlineLoop class="h-16 w-16" />
                <div class="text-center">
                    <label for="file_upload" class="font text-2xl font-semibold">Upload CSV</label>
                    <h3 class="font-light text-text-secondary">Drag or click to upload a file</h3>
                </div>
            {/if}

        </div>
    </div>

</div>