<script lang="ts">
  import Container from './ui/Container.svelte';

  const today = new Date();
</script>

<footer class="border-t border-gray-200">
  <Container>
    <p class="text-center text-base text-gray-600">
      &copy; {today.getFullYear()} Evgeny Nizhibitsky.
    </p>
  </Container>
</footer>
