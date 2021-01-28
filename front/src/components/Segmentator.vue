<template>
  <div class="grid grid-cols-2 gap-8 py-8 px-4 mx-auto max-w-7xl md:grid-cols-3 sm:px-6 lg:px-8">
    <div class="flex col-span-2 mt-1 rounded-md shadow-sm">
      <div class="flex relative flex-grow items-stretch focus-within:z-10">
        <label for="photo" class="sr-only">Email</label>
        <input
          v-model="photo"
          placeholder="http://example.com/image.png"
          type="text"
          id="photo"
          class="block w-full rounded-none rounded-l-md border-gray-300 focus:ring-blue-500 focus:border-blue-500"
        >
      </div>
      <button
        :disabled="!photo || !ready"
        class="inline-flex relative justify-center items-center py-2 px-4 -ml-px space-x-2 w-24 h-12 font-medium text-white bg-blue-600 rounded-r-md border border-blue-700 hover:bg-blue-700 focus:outline-none focus:ring-1 focus:ring-blue-500 focus:border-blue-500"
        @click="call"
      >
        <span v-if="ready">
          Poehali!
        </span>
        <IconTailSpin v-if="!ready" class="w-6 h-6" />
      </button>
    </div>

    <div class="overflow-hidden col-span-2 bg-white sm:rounded-lg sm:shadow sm:col-span-1 sm:col-start-1">
      <div class="flex flex-col py-4 px-3 space-y-4 bg-white sm:px-4">
        <h3 class="text-3xl font-medium leading-6 text-gray-900">
          Photo
        </h3>
        <figure class="aspect-w-3 aspect-h-4">
          <img :src="preview" alt="Photo">
        </figure>
      </div>
    </div>

    <div
      v-if="mask || error"
      class="overflow-hidden col-span-2 sm:rounded-lg sm:shadow sm:col-span-1"
      :class="{'bg-red-50 text-red-600': error, 'bg-white text-gray-900': mask}"
    >
      <div class="flex flex-col py-4 px-3 space-y-4 sm:px-4">
        <h3 class="text-3xl font-medium leading-6">
          <span v-if="mask">Mask</span>
          <span v-else>Error</span>
        </h3>
        <img v-if="mask" :src="'data:image/png;base64,' + mask" alt="Mask">
        <pre v-else class="whitespace-pre-wrap"><code>{{ error }}</code></pre>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, reactive, computed, watch } from 'vue'

import useWebAPI from '@/composables/useWebAPI'

import IconTailSpin from '@/components/IconTailSpin'

export default {
  components: {
    IconTailSpin
  },
  setup () {
    const photo = ref('')
    const index = Math.floor(Math.random() + 1.5).toString()
    photo.value = window.location.origin + '/images/sample' + index + '.jpg'

    const preview = computed(() => photo.value || '//placeimg.com/480/640/any')
    const request = reactive({ url: photo })

    const { ready, error, result, call } = useWebAPI('api/segment', request)
    const mask = computed(() => result.value && result.value.data && result.value.data.mask)

    watch(photo, () => { result.value = '' })

    return {
      photo,
      preview,
      ready,
      error,
      mask,
      call
    }
  }
}
</script>
