<script lang="ts">
  import Card from './ui/Card.svelte';

  interface Props {
    mask: string;
  }

  let { mask }: Props = $props();
  let src = $derived(`data:image/png;base64,${mask}`);
</script>

<Card>
  <h3 class="text-3xl font-medium leading-6 text-gray-900">Mask</h3>
  <img {src} class="min-w-full max-w-fit border border-gray-500" alt="Generated Mask" />
</Card>
