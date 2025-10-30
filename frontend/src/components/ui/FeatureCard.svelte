<script lang="ts">
    let {title, desc} = $props()
</script>

<div class="rounded-lg border border-dashed border-border-primary bg-primary px-4 py-3 max-w-96">
    <h3 class="text-text-primary text-lg font-semibold">{title}</h3>
    <p class="text-text-secondary">{desc}</p>
</div>