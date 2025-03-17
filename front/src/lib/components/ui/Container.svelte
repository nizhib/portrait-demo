<script lang="ts">
  import clsx from 'clsx';

  export let noxspx = false;
  let className = '';
  export { className as class };
</script>

<div
  class={clsx(
    'mx-auto max-w-7xl py-4 sm:px-6 sm:py-6 lg:px-8 lg:py-8',
    noxspx || 'px-4',
    className,
  )}
>
  <slot />
</div>
