<script lang="ts">
  import { image } from '$lib/store.svelte';
  import IconTailSpin from './IconTailSpin.svelte';

  interface Props {
    onStart: () => void;
    onFetch: (data: any) => void;
    onError: (message: string) => void;
  }

  let { onStart, onFetch, onError }: Props = $props();
  let isLoading = $state(false);

  async function handleSubmit() {
    isLoading = true;
    onStart();
    try {
      const response = await fetch('/api/segment', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url: image.url }),
      });
      if (!response.ok) {
        let message = await response.text();
        try {
          message = JSON.parse(message).message;
        } catch {
          // ignore
        } finally {
          message ||= `${response.status}: ${response.statusText}`;
          onError(message);
        }
      } else {
        const data = await response.json();
        onFetch(data);
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      onError(message);
    } finally {
      isLoading = false;
    }
  }
</script>

<div class="flex shadow-sm sm:col-span-2 sm:rounded-md">
  <div class="relative flex flex-grow items-stretch focus-within:z-10">
    <label for="url" class="sr-only">Photo</label>
    <input
      id="url"
      type="text"
      bind:value={image.url}
      class="block w-full rounded-none border-gray-300 focus:border-blue-500 focus:ring-blue-500 sm:rounded-l-md"
      placeholder="https://example.com/photo.jpg"
    />
  </div>
  <button
    class="relative -ml-px inline-flex h-12 w-24 items-center justify-center space-x-2 border border-blue-700 bg-blue-600 px-4 py-2 font-medium text-white focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500 enabled:hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50 sm:rounded-r-md"
    disabled={!image.url || isLoading}
    onclick={handleSubmit}
    type="submit"
  >
    {#if isLoading}
      <IconTailSpin class="h-6 w-6" />
    {:else}
      <span>Poehali!</span>
    {/if}
  </button>
</div>
