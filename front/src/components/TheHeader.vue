<template>
  <header class="bg-blue-600 shadow">
    <div class="py-8 px-4 mx-auto max-w-7xl sm:px-6 lg:px-8">
      <h1 class="text-4xl font-bold leading-tight text-white">
        {{ title }}
      </h1>
    </div>
  </header>
</template>

<script>
export default {
  props: {
    title: {
      type: String,
      default: 'TheHeader'
    }
  }
}
</script>
