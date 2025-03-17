<script lang="ts">
  import Container from './ui/Container.svelte';

  interface Props {
    title: string;
  }

  let { title }: Props = $props();
</script>

<header class="bg-blue-600 shadow">
  <Container>
    <h1 class="text-3xl font-bold leading-tight text-white sm:text-4xl">
      {title}
    </h1>
  </Container>
</header>
