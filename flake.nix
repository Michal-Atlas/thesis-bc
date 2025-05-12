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
        {
          self',
          pkgs,
          system,
          ...
        }:
        {
          #          _module.args.pkgs = import inputs.nixpkgs {
          #           inherit system;
          #           config = {
          #cudaSupport = true;
          #allowBroken = true;
          #allowUnfree = true;
          #          };
          #        };
          treefmt = {
            projectRootFile = "flake.nix";
            programs = {
              nixfmt.enable = true;
              keep-sorted.enable = true;
            };
          };
          packages = {
            modern = pkgs.stdenvNoCC.mkDerivation {
              name = "optex-modern";
              src = pkgs.fetchurl {
                url = "https://git.sr.ht/~michal_atlas/gtex/blob/master/src/modern.tex";
                hash = "sha256-UsrobEY6bczDZHWpQ41yzfYJCyccaQew1zVzfPsEEVA";
              };
              dontUnpack = true;
              installPhase = ''
                dest=$out/tex/luatex/modern
                mkdir -p $dest
                cp $src $dest/modern.tex
              '';
              passthru.tlType = "run";
            };
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
              inherit (pkgs.texlive)
                scheme-small
                optex
                firamath
                xits
                ;
              ctustyle3 = {
                pkgs = with self'.packages; [
                  ctustyle3
                  modern
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
                ml-dtypes
                numpy
                onnx
                onnxruntime
                torch
                typing-extensions
                # keep-sorted end
              ];
            };
            apache-tvm = pkgs.tvm.overrideAttrs (super: {
              version = "0.19.0";
            });
            python-apache-tvm =
              let
                super = self'.packages.apache-tvm;
              in
              pkgs.python311Packages.buildPythonPackage
                #pkgs.stdenv.mkDerivation
                {
                  pname = "tvm";
                  inherit (super) version;
                  format = "pyproject";
                  src = super.src;
                  preUnpack = ''
                    cp -r $src work
                    chmod -R 777 work
                    sed 's|INPLACE_BUILD = .*|INPLACE_BUILD = True|g' -i work/python/setup.py
                  '';
                  sourceRoot = "work/python";
                  TVM_LIBRARY_PATH = "${super}/lib";
                  dependencies = with pkgs.python311Packages; [
                    attrs
                    cloudpickle
                    decorator
                    ml-dtypes
                    numpy
                    psutil
                    scipy
                    tornado
                    typing-extensions
                  ];
                  buildInputs = [
                    super
                  ];
                  nativeBuildInputs = with pkgs; [
                    python311Packages.setuptools
                    git
                  ];
                };
          };
          devShells.default = pkgs.mkShell {
            nativeBuildInputs = with pkgs; [
              (pkgs.writeShellScript "set-tvm-var" ''
                export TVM_LIBRARY_PATH="${pkgs.lib.makeLibraryPath [ self'.packages.apache-tvm ]}"; 
              '')
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
                  keras
                  self'.packages.onnxscript
                  self'.packages.python-apache-tvm
                  tensorflow
                  # tensorflow-datasets
                  torch
                  # keep-sorted end
                ];
                ignoreCollisions = true;
              })
              (pkgs.writeShellScriptBin "connect" ''
                ${pkgs.minicom}/bin/minicom --device /dev/ttyUSB3 --baudrate 115200
              '')
              (pkgs.writeShellScriptBin "compenv" ''
                docker run --rm -it -v /blackpool/thesis/work/scarthgap.TQ.ARM.BSP.0001/:/src -v $PWD/meta-overlay:/src/ci-meta-tq/sources/meta-overlay:ro -v $PWD/conf:/src/ci-meta-tq/tqma8mpxl_build/conf:ro --userns=keep-id thesis-bitbake
              '')
              (pkgs.writeShellScriptBin "prepare" ''
                docker run --rm -it -v ./py:/root thesis-py
              '')
              (pkgs.writeShellScriptBin "transfer" ''
                scp -r py/test_model* py/test_rig imx:
              '')
              (pkgs.writeShellScriptBin "rebib" ''
                papis --cc --color always export --all > ~/cl/thesis/thesis.bib
              '')
            ];
          };
        };
    };
}
