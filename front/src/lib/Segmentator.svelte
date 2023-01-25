<script lang="ts">
  import { onMount } from 'svelte';
  import Container from './ui/Container.svelte';
  import RequestForm from './RequestForm.svelte';
  import PhotoCard from './PhotoCard.svelte';
  import MaskCard from './MaskCard.svelte';
  import ErrorCard from './ErrorCard.svelte';

  type HavingMask = {
    mask?: string;
  };

  type HavingData = {
    data?: HavingMask;
  };

  const placeholder = 'https://placeimg.com/480/640/any';
  let url = '';
  let data: HavingData = {};
  let error = '';

  onMount(() => {
    const origin =
      typeof window !== 'undefined' && window.location.origin
        ? window.location.origin
        : '';
    const sample1 = `${origin}/images/sample1.jpg`;
    const sample2 = `${origin}/images/sample2.jpg`;
    url = Math.random() > 0.5 ? sample1 : sample2;
  });

  function reset() {
    data = {};
    error = '';
  }

  $: url, reset();
  $: mask = data?.data?.mask;

  function handleFetch(dataEvent: CustomEvent<HavingData>) {
    data = dataEvent.detail;
  }

  function handleError(errorEvent: CustomEvent<string>) {
    error = errorEvent.detail;
  }
</script>

<Container noxspx class="grid gap-4 sm:grid-cols-2 sm:gap-6 lg:gap-8 xl:grid-cols-3">
  <RequestForm
    bind:url
    {placeholder}
    on:start={reset}
    on:fetch={handleFetch}
    on:error={handleError}
  />
  <PhotoCard src={url || placeholder} />
  {#if mask}
    <MaskCard {mask} />
  {/if}
  {#if error}
    <ErrorCard {error} />
  {/if}
</Container>
