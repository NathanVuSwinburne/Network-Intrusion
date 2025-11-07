<script lang="ts">
    const { name, id, options, errors } = $props();

    const error = $derived(errors[id] != undefined)
</script>

<div class="flex flex-col">
    <label class="text-text-primary" for={id}>{name}</label>

    {#if error}
        <label class="text-red-800 text-xs" for={id}>{errors[id]}</label>
    {/if}

    <select id={id} name={id} class={`text-text-primary rounded border py-1 px-2 ${error ? 'border-red-800' : 'border-border-primary'}`}>
        {#each options as option}
            <option value={option} class="text-black">{option}</option>
        {/each}
    </select>
</div>