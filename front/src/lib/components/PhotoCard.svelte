<script lang="ts">
  import { image } from '$lib/store.svelte';
  import Card from './ui/Card.svelte';

  let { width, height, alt } = image;
  let src = $derived(image.url || 'https://placehold.co/480x640');
</script>

<Card class="col-start-1">
  <h2 class="text-3xl font-medium leading-6 text-gray-900">Photo</h2>
  <img class="min-w-full border border-white" {src} {width} {height} {alt} />
</Card>
