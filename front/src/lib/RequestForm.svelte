<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import IconTailSpin from './IconTailSpin.svelte';

  export let placeholder: string;
  export let url: string;
  let loading = false;

  const dispatch = createEventDispatcher();

  async function handleSubmit() {
    loading = true;
    dispatch('start');
    try {
      const response = await fetch('/api/segment', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url }),
      });
      if (!response.ok) {
        let message = await response.text();
        try {
          message = JSON.parse(message).message;
        } catch {
          // ignore
        } finally {
          message ||= `${response.status}: ${response.statusText}`;
          dispatch('error', message);
        }
      } else {
        const data = await response.json();
        dispatch('fetch', data);
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      dispatch('error', message);
    } finally {
      loading = false;
    }
  }
</script>

<div class="flex shadow-sm sm:col-span-2 sm:rounded-md">
  <div class="relative flex flex-grow items-stretch focus-within:z-10">
    <label for="url" class="sr-only">Photo</label>
    <input
      id="url"
      type="text"
      bind:value={url}
      class="block w-full rounded-none border-gray-300 focus:border-blue-500 focus:ring-blue-500 sm:rounded-l-md"
      {placeholder}
    />
  </div>
  <button
    class="relative -ml-px inline-flex h-12 w-24 items-center justify-center space-x-2 border border-blue-700 bg-blue-600 px-4 py-2 font-medium text-white focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500 enabled:hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50 sm:rounded-r-md"
    disabled={!url || loading}
    on:click={handleSubmit}
    type="submit"
  >
    {#if loading}
      <IconTailSpin class="h-6 w-6" />
    {:else}
      <span>Poehali!</span>
    {/if}
  </button>
</div>
