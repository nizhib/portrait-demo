<script lang="ts">
  import Card from './ui/Card.svelte';

  interface Props {
    error: string;
  }

  let { error }: Props = $props();
</script>

<Card>
  <h3 class="text-3xl font-medium leading-6 text-red-600">Error</h3>
  <pre class="whitespace-pre-wrap"><code>{error.toString()}</code></pre>
</Card>
