{
  description = "Atlas' NixOS configuration";

  inputs = {
    nixpkgs.url = "github:NixOs/nixpkgs/nixos-24.11";
    flake-parts.url = "github:hercules-ci/flake-parts";
    systems.url = "github:nix-systems/default";
  };

  outputs =
    {
      flake-parts,
      systems,
      pre-commit-hooks,
      nixos-flake,
      ...
    }@inputs:
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = import systems;
      flake.lib = import ./lib;
      perSystem =
        { self', pkgs, ... }:
        {
          packages.ctustyle3 = pkgs.stdenvNoCC.mkDerivation {
            name = "CTUstyle3";
            src = pkgs.fetchFromGitHub {
              owner = "olsak";
              repo = "CTUstyle3";
              rev = "cae0d839418f79dffa7ba80d5951962ef090c12c";
              hash = "sha256-quh74CKfeVo/UHeF0hvlLtz+wzQQAPdAlRJP7m6rs78=";
            };
            outputs = [
              "out"
              "texdoc"
            ];
            installPhase = ''
              dest=$out/tex/luatex/ctustyle3
              mkdir -p $dest
              cp $src/ctustyle3.tex $src/logo_* $src/ctulogo-* $dest/
              dest=$out/fonts/opentype/ctu/Technika
              mkdir -p $dest
              cp $src/*.otf $dest/
              cp $src/ctustyle-doc.pdf $texdoc
            '';
            passthru.tlType = "run";
          };
          devShells.default = pkgs.mkShell {
            nativeBuildInputs = with pkgs; [
              self'.packages.ctustyle3
              (texlive.combine {
                inherit (texlive) scheme-small optex;
                ctustyle3 = {
                  pkgs = [
                    self'.packages.ctustyle3
                  ];
                };
              })
            ];

          };
        };
    };
}
