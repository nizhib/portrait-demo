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

  let data: HavingData = {};
  let error = '';

  const placeholder = 'https://placeimg.com/480/640/any';
  let url = '';
  let width: number = undefined;
  let height: number = undefined;

  const samples = [
    { url: '/images/sample1.webp', width: 597, height: 800 },
    { url: '/images/sample2.webp', width: 450, height: 657 },
  ];
  ({ url, width, height } = samples[Math.floor(Math.random() * samples.length)]);

  onMount(() => (url = `${window.location.origin}${url}`));
  $: photoUrl = url || placeholder;

  function reset() {
    if (!url.startsWith('/') && !url.startsWith(window.location.origin)) {
      width = undefined;
      height = undefined;
    }
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
  <PhotoCard src={photoUrl} {width} {height} />
  {#if mask}
    <MaskCard {mask} />
  {/if}
  {#if error}
    <ErrorCard {error} />
  {/if}
</Container>
