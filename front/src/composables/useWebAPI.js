import { ref } from 'vue'

export default function useWebAPI (apiRoot, request) {
  const ready = ref(true)
  const error = ref('')
  const result = ref('')

  const call = async () => {
    ready.value = false
    error.value = ''
    result.value = ''

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
        result.value = JSON.parse(text)
      } catch (e) {
        console.log(response)
        error.value = 'The response is not JSON: ' + text
      }
    } else {
      try {
        const data = await response.json()
        error.value = data.message || data
      } catch (e) {
        error.value = response.status + ' ' + response.statusText
      }
    }

    ready.value = true
  }

  return {
    ready,
    error,
    result,
    call
  }
}
