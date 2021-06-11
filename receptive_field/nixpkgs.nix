let
  # now tracking https://github.com/tbenst/nixpkgs/tree/nix-data
  # when updating, replace all in project
  nixpkgsSHA = "cd63096d6d887d689543a0b97743d28995bc9bc3";
  pkgs = import (fetchTarball
    "https://github.com/NixOS/nixpkgs/archive/${nixpkgsSHA}.tar.gz") {
      system = builtins.currentSystem;
    };

in pkgs