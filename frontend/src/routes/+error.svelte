<script>
    import { page } from '$app/stores';
    import { goto } from '$app/navigation';
    import { onMount } from 'svelte';

    onMount(() => {
        if ($page.status === 404) {
            goto('/', { replaceState: true });
        }
    });
</script>

{#if $page.status === 404}
    <p>Redirecting to home...</p>
{:else}
    <h1>{$page.status}: {$page.error?.message}</h1>
{/if}