<script lang="ts">
  import Card from './ui/Card.svelte';

  export let src: string = undefined;
  export let width: number = undefined;
  export let height: number = undefined;
</script>

<Card class="col-start-1">
  <h2 class="text-3xl font-medium leading-6 text-gray-900">Photo</h2>
  <img {src} {width} {height} class="min-w-full border border-white" alt="Source" />
</Card>
