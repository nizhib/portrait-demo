<template>
  <div
    id="app"
    class="flex flex-col min-h-screen bg-gray-100"
  >
    <TheHeader title="Portrait Segmentation" />
    <main class="flex-1">
      <Segmentator />
    </main>
    <TheFooter />
  </div>
</template>

<script>
import TheHeader from '@/components/TheHeader'
import Segmentator from '@/components/Segmentator'
import TheFooter from '@/components/TheFooter'

export default {
  name: 'PortraitDemo',
  components: {
    TheHeader,
    Segmentator,
    TheFooter
  }
}
</script>
