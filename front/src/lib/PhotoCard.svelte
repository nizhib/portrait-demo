<script lang="ts">
  import Card from './ui/Card.svelte';

  export let src: string;
</script>

<Card class="col-start-1">
  <h2 class="text-3xl font-medium leading-6 text-gray-900">Photo</h2>
  <img
    {src}
    class="min-w-full max-w-fit border border-white"
    alt="Source"
    loading="lazy"
  />
</Card>
