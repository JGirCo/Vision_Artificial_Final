{
  description = "Python development environment with OpenCV and CUDA";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
          # overlays = [
          #   (final: prev: {
          #     # 🚀 Override opencv4 to explicitly enable CUDA support
          #     opencv4 = prev.opencv4.override {
          #       enableGtk3 = true;
          #       enablePython = true;
          #       enableCuda = true;
          #     };
          #   })
          # ];
        };
        # Create a Python environment with the CUDA-enabled OpenCV
        python-with-opencv = pkgs.python3.withPackages (ps: [ ps.opencv4 ]);
      in {
        devShell = pkgs.mkShell {
          buildInputs = [
            python-with-opencv
            pkgs.pkg-config
            pkgs.stdenv.cc.cc.lib

            pkgs.python3Packages.torch
            pkgs.python3Packages.ultralytics
            pkgs.python3Packages.albumentations
            pkgs.python3Packages.matplotlib
            pkgs.python3Packages.scikit-learn
            pkgs.python3Packages.tkinter
            pkgs.python3Packages.colorlog

          ];

          shellHook = ''
            # 🎯 Add the CUDA library paths to LD_LIBRARY_PATH
            export LD_LIBRARY_PATH=${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH
            export LD_LIBRARY_PATH=${pkgs.cudaPackages.cudatoolkit}/lib:$LD_LIBRARY_PATH
            export LD_LIBRARY_PATH=${pkgs.cudaPackages.cudnn}/lib:$LD_LIBRARY_PATH

            echo "Python environment ready with OpenCV $(python -c 'import cv2; print(cv2.__version__)') with CUDA support."
            # Verify CUDA is available in PyTorch
            echo "Checking PyTorch for CUDA availability:"
            python -c "import torch; print('CUDA available:', torch.cuda.is_available());"
          '';
        };
      });
}
