## To build
```
> cd receptive_field
> nix-shell shell.nix
$ yarn bundle
$ cp dist/*.js ..
$ cp dist/*.html ..
```

Requires images in pixel2retina and retina2pixel folders. To run locally, for instance because you want to visualize your own data, follow instructions at https://github.com/tbenst/glia/blob/gh-pages/receptive_field/README.md#hot-reloading. I assume that the model uses only 32 x 32 channels and the image is 64 x 64 channels, but this can be easily changed by modifying the number i.e. [here](https://github.com/tbenst/glia/blob/7a857d4ec8d1275aa1f3f4a6077a420768cc4c56/receptive_field/src/Retina2pixel.purs#L27).
