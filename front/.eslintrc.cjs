module.exports = {
  root: true,
  env: {
    browser: true,
    es2021: true,
  },
  extends: [
    'airbnb-base',
    'plugin:import/recommended',
    'plugin:@typescript-eslint/recommended',
    'plugin:prettier/recommended',
  ],
  overrides: [
    {
      files: ['*.svelte'],
      processor: 'svelte3/svelte3',
      rules: {
        'import/first': 0,
        'import/no-mutable-exports': 0,
        'import/prefer-default-export': 0,
        'import/no-extraneous-dependencies': 0,
      },
    },
  ],
  parser: '@typescript-eslint/parser',
  plugins: ['svelte3', 'import', 'prettier', '@typescript-eslint'],
  rules: {
    // prettier
    'prettier/prettier': 'error',

    'prefer-const': 'error',
    '@typescript-eslint/no-unused-vars': 'error',
    '@typescript-eslint/no-explicit-any': 'error',
  },
  settings: {
    'svelte3/typescript': 2,
  },
};
