import Vue from 'vue'
import App from '@/App.vue'
import '@/main.css'

Vue.config.productionTip = false
Vue.config.devtools = true

new Vue({
  ...App
}).$mount('#app')
