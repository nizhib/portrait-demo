<script lang="ts">
  let className = '';

  export { className as class };
</script>

<div class={`overflow-hidden bg-white shadow-sm sm:rounded-lg sm:shadow ${className}`}>
  <div class="px-4 py-5 sm:p-6">
    <div class="flex flex-col space-y-5 sm:space-y-6">
      <slot />
    </div>
  </div>
</div>
