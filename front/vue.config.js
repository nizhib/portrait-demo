module.exports = {
  css: {
    sourceMap: true
  },
  chainWebpack: config => {
    config
      .plugin('html')
      .tap(args => {
        args[0].title = 'Portrait Segmentation'
        return args
      })
  },
  devServer: {
    proxy: {
      '^/api/': {
        target: 'http://127.0.0.1:5000',
        pathRewrite: { '^/api/': '' },
        secure: false
      }
    }
  }
}
