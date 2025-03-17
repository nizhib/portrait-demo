<script lang="ts">
  import type { Snippet } from 'svelte';

  interface Props {
    class?: string;
    children: Snippet;
  }

  let { class: className, children }: Props = $props();
</script>

<div class={`overflow-hidden bg-white shadow-sm sm:rounded-lg sm:shadow ${className}`}>
  <div class="px-4 py-5 sm:p-6">
    <div class="flex flex-col space-y-5 sm:space-y-6">
      {@render children?.()}
    </div>
  </div>
</div>
