<script lang="ts">
  import { image } from '$lib/store.svelte';
  import Container from './ui/Container.svelte';
  import ErrorCard from './ErrorCard.svelte';
  import MaskCard from './MaskCard.svelte';
  import PhotoCard from './PhotoCard.svelte';
  import RequestForm from './RequestForm.svelte';

  type HavingMask = {
    mask?: string;
  };

  type HavingData = {
    data?: HavingMask;
  };

  let result: HavingData = $state({});
  let mask = $derived(result?.data?.mask);
  let error = $state('');

  let prevUrl = $state(image.url);

  $effect(() => {
    if (image.url.startsWith('/')) {
      image.url = `${window.location.origin}${image.url}`;
      prevUrl = image.url;
    }
  });

  $effect(() => {
    if (image.url !== prevUrl) {
      prevUrl = image.url;
      image.width = null;
      image.height = null;
      image.alt = null;
    }
  });

  function resetResult() {
    result = {};
    error = '';
  }

  function handleFetch(resultValue: HavingData) {
    result = resultValue;
  }

  function handleError(errorValue: string) {
    error = errorValue;
  }
</script>

<Container noxspx class="grid gap-4 sm:grid-cols-2 sm:gap-6 lg:gap-8 xl:grid-cols-3">
  <RequestForm onStart={resetResult} onFetch={handleFetch} onError={handleError} />
  <PhotoCard />
  {#if mask}
    <MaskCard {mask} />
  {/if}
  {#if error}
    <ErrorCard {error} />
  {/if}
</Container>
