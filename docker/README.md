# How to build Dockerfile?

```shell
docker build --tag torch:dev --file Dockerfile .
```

# 開発時の注意点

- Working Directoryは`torch-model`
- PyCharmの`Edit Configuration`から，Docker runのオプションに`--gpus all`を追加し，Working Directoryを`torch-model`に直す

