<script lang="ts">
    const { name, id } = $props();
</script>

<div class="flex flex-col">
    <label class="text-text-primary" for={id}>{name}</label>
    <input type="text" id={id} class="text-text-primary rounded border border-border-primary py-1 px-2 w-min">
</div>