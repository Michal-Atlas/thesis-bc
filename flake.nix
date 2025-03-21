{
  description = "Atlas' NixOS configuration";

  inputs = {
    nixpkgs.url = "github:NixOs/nixpkgs/nixos-24.11";
    flake-parts.url = "github:hercules-ci/flake-parts";
    systems.url = "github:nix-systems/default";
    treefmt.url = "github:numtide/treefmt-nix";
  };

  outputs =
    {
      flake-parts,
      systems,
      treefmt,
      ...
    }@inputs:
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = import systems;
      flake.lib = import ./lib;
      imports = [ treefmt.flakeModule ];
      perSystem =
        { self', pkgs, ... }:
        {
          treefmt = {
            projectRootFile = "flake.nix";
            programs = {
              nixfmt.enable = true;
              keep-sorted.enable = true;
            };
          };
          packages = {
            ctustyle3 = pkgs.stdenvNoCC.mkDerivation {
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
            tex = pkgs.texlive.combine {
              inherit (pkgs.texlive) scheme-small optex;
              ctustyle3 = {
                pkgs = [
                  self'.packages.ctustyle3
                ];
              };
            };
            default = pkgs.stdenvNoCC.mkDerivation {
              name = "thesis.pdf";
              src = ./.;
              dontUnpack = true;
              allowSubstitutes = false;
              buildPhase = ''
                export HOME=$PWD
                cp -r $src/* .
                while ${self'.packages.tex}/bin/optex thesis.tex | tee /dev/stderr | grep -Eq 'again|rerun'; do :; done
              '';
              installPhase = ''
                mv *.pdf $out
              '';
            };
            onnxscript = pkgs.python311Packages.buildPythonPackage rec {
              pname = "onnxscript";
              version = "0.2.0";
              format = "pyproject";
              src = pkgs.fetchPypi {
                inherit pname version;
                hash = "sha256-H0UGxTPBfxlrukZTIU9mbBrgBh+ffsoFMY93L25Eut4=";
              };
              build-system = with pkgs.python311Packages; [ setuptools ];
              nativeBuildInputs = with pkgs; [ git ];
              dependencies = with pkgs.python311Packages; [
                # keep-sorted start
                torch
                numpy
                onnx
                typing-extensions
                ml-dtypes
                # keep-sorted end
              ];
            };
          };
          devShells.default = pkgs.mkShell {
            nativeBuildInputs = with pkgs; [
              # keep-sorted start
              bmaptool
              git-repo
              minicom
              multipath-tools
              self'.packages.ctustyle3
              self'.packages.tex
              uuu
              # keep-sorted end
              (python311.buildEnv.override {
                extraLibs = with python311Packages; [
                  # keep-sorted start
                  tensorflow
                  torch
                  self'.packages.onnxscript
                  # keep-sorted end
                ];
                ignoreCollisions = true;
              })
              (pkgs.writeShellScriptBin "connect" ''
                ${pkgs.minicom}/bin/minicom --device /dev/ttyUSB3 --baudrate 115200
              '')
              (pkgs.writeShellScriptBin "watch" ''
                ${pkgs.entr}/bin/entr -s 'nix build . -o thesis.pdf' <<< thesis.tex
              '')
              (pkgs.writeShellScriptBin "bind-mounts" ''
                sudo mount --bind meta-overlay /blackpool/thesis/work/scarthgap.TQ.ARM.BSP.0001/ci-meta-tq/sources/meta-overlay
                sudo mount --bind conf /blackpool/thesis/work/scarthgap.TQ.ARM.BSP.0001/ci-meta-tq/tqma8mpxl_build/conf
              '')
              (pkgs.writeShellScriptBin "compenv" ''
                docker run --rm -it -v /blackpool/thesis/work/scarthgap.TQ.ARM.BSP.0001/:/src --userns=keep-id 26d0
              '')
            ];
          };
        };
    };
}
