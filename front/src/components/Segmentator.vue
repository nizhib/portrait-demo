<template>
  <div class="grid grid-cols-2 gap-8 py-8 px-4 mx-auto max-w-7xl md:grid-cols-3 sm:px-6 lg:px-8">
    <div class="flex col-span-2 mt-1 rounded-md shadow-sm">
      <div class="flex relative flex-grow items-stretch focus-within:z-10">
        <label for="url" class="sr-only">Email</label>
        <input
          v-model="url"
          placeholder="http://example.com/image.png"
          type="text"
          id="url"
          class="block w-full rounded-none rounded-l-md border-gray-300 focus:ring-blue-500 focus:border-blue-500"
        >
      </div>
      <button
        :disabled="!url || state !== 'ready'"
        class="inline-flex relative justify-center items-center py-2 px-4 -ml-px space-x-2 w-24 h-12 font-medium text-white bg-blue-600 rounded-r-md border border-blue-700 hover:bg-blue-700 focus:outline-none focus:ring-1 focus:ring-blue-500 focus:border-blue-500"
        @click="segment"
      >
        <span v-if="state === 'ready'">
          Poehali!
        </span>
        <IconTailSpin v-if="state === 'segment'" class="w-6 h-6" />
      </button>
    </div>

    <div class="overflow-hidden col-span-2 bg-white sm:rounded-lg sm:shadow sm:col-span-1 sm:col-start-1">
      <div class="flex flex-col py-4 px-3 space-y-4 h-full bg-white sm:px-4">
        <h3 class="text-3xl font-medium leading-6 text-gray-900">
          Photo
        </h3>
        <img v-if="url" class="h-full" :src="url" alt="Image">
        <img v-else class="h-full" src="//placeimg.com/480/640/any" alt="Image">
      </div>
    </div>

    <div v-if="mask || error" class="overflow-hidden col-span-2 bg-white sm:rounded-lg sm:shadow sm:col-span-1">
      <div class="flex flex-col py-4 px-3 space-y-4 h-full bg-white sm:px-4">
        <h3 class="text-3xl font-medium leading-6 text-gray-900">
          <span v-if="mask">Mask</span>
          <span v-else>Error</span>
        </h3>
        <img v-if="mask" class="h-full" :src="'data:image/png;base64,' + mask" alt="Mask">
        <pre v-else><code>{{ error }}</code></pre>
      </div>
    </div>
  </div>
</template>

<script>
import IconTailSpin from '@/components/IconTailSpin'

export default {
  components: {
    IconTailSpin
  },
  data () {
    return {
      url: '',
      state: 'ready',
      error: '',
      mask: ''
    }
  },
  methods: {
    clear () {
      this.error = ''
      this.mask = ''
    },
    async segment () {
      this.clear()
      this.state = 'segment'

      const request = { url: this.url }
      const response = await fetch('api/segment', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(request)
      })

      if (response.ok) {
        const text = await response.text()
        try {
          const result = JSON.parse(text)
          this.mask = result.data.mask
        } catch (e) {
          console.log(response)
          this.error = 'The response is not JSON: ' + text
        }
      } else {
        try {
          const error = await response.json()
          this.error = error.message || error
        } catch (e) {
          this.error = response.status + ' ' + response.statusText
        }
      }

      this.state = 'ready'
    }
  },
  mounted () {
    const index = Math.floor(Math.random() + 1.5).toString()
    this.url = window.location.origin + '/images/sample' + index + '.jpg'
  }
}
</script>
