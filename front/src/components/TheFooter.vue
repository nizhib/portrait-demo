<template>
  <footer class="w-full border-t border-gray-200">
    <div class="overflow-hidden py-8 px-4 mx-auto max-w-7xl sm:px-6 lg:px-8">
      <p class="text-base text-center text-gray-400">
        &copy; {{ year }} Evgeny Nizhibitsky.
      </p>
    </div>
  </footer>
</template>

<script>
export default {
  setup () {
    const today = new Date()

    return {
      year: today.getFullYear()
    }
  }
}
</script>
