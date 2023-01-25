<script lang="ts">
  import Header from './lib/Header.svelte';
  import Segmentator from './lib/Segmentator.svelte';
  import Footer from './lib/Footer.svelte';
</script>

<div class="flex min-h-screen flex-col bg-gray-100">
  <Header title="Portrait Segmentation" />
  <main class="flex-1">
    <Segmentator />
  </main>
  <Footer />
</div>
