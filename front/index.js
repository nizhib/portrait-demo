Vue.directive("highlightjs", {
    deep: true,
    bind: function (el, binding) {
        // on first bind, highlight all targets
        const targets = el.querySelectorAll("code");
        targets.forEach(function (target) {
            // if a value is directly assigned to the directive,
            // use this instead of the element content.
            if (binding.value) {
                target.textContent = binding.value
            }
            // noinspection JSUnresolvedVariable
            hljs.highlightBlock(target);
        });
    },
    componentUpdated: function (el, binding) {
        // after an update, re-fill the content and then highlight
        const targets = el.querySelectorAll("code");
        targets.forEach(function (target) {
            if (binding.value) {
                target.textContent = binding.value;
                // noinspection JSUnresolvedVariable
                hljs.highlightBlock(target);
            }
        });
    }
});

if (!window.location.origin) {
    window.location.origin = window.location.protocol +
        "//" + window.location.hostname +
        (window.location.port ? ":" + window.location.port: "");
}

const app = new Vue({
    el: "#app",
    template: "#portrait",
    data: {
        date: 2020,
        endpoint: window.location.origin + "/api/",
        url: "",
        preview: true,
        state: "ready",
        error: "",
        mask: ""
    },
    created() {
        const today = new Date();
        this.date = today.getFullYear();
    },
    methods: {
        clear() {
            this.error = "";
            this.mask = "";
        },
        segment() {
            const vm = this;

            vm.clear();
            vm.state = "segment";

            axios.post(vm.endpoint + "segment", {
                url: vm.url
            })
            .then(function (response) {
                const result = response.data;

                if (result["success"]) {
                    vm.mask = result["data"]["mask"];
                } else {
                    vm.error = result["message"];
                    if (!vm.error) {
                        vm.error = "Unexpected Error";
                    }
                }
                vm.state = "ready";
            })
            .catch(function (error) {
                vm.error = error;
                if (!vm.error) {
                    vm.error = "Unexpected Error";
                }
                vm.state = "ready";
            });
        }
    },
    mounted() {
        const index = Math.floor(Math.random() + 1.5).toString();
        this.url = window.location.origin + "/assets/images/sample" + index + ".jpg";
    }
});
