{
  description = "Implementation of 3D Gaussian Ray Tracing: Fast Tracing of Particle Scenes";

  inputs =
    {
      nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
      flake-utils.url = "github:numtide/flake-utils";
    };

  outputs =
    { self
    , nixpkgs
    , flake-utils
    , ...
    }:
      with flake-utils.lib;
      eachSystem [
        system.x86_64-linux
        system.x86_64-darwin
        system.aarch64-darwin
      ]
        (system:
        let
          inherit (nixpkgs) lib;
          pkgs = import nixpkgs {
            inherit system;
            config.allowUnfree = true;
            config.cudaSupport = true;
            config.cudaVersion = "12";
          };

          # Use Python 3.12 from nixpkgs
          python = pkgs.python312;

          buildInputs = with pkgs; [
            # Python environment.
            python
            uv
          ];

          env = {
            # Prevent uv from managing Python downloads
            UV_PYTHON_DOWNLOADS = "never";
            # Force uv to use nixpkgs Python interpreter
            UV_PYTHON = python.interpreter;
          }
          // lib.optionalAttrs pkgs.stdenv.isLinux {
            # Python libraries often load native shared objects using dlopen(3).
            # Setting LD_LIBRARY_PATH makes the dynamic library loader aware of libraries without using RPATH for lookup.
            LD_LIBRARY_PATH = lib.makeLibraryPath pkgs.pythonManylinuxPackages.manylinux1;
          };
        in
        {
          devShells.default = pkgs.mkShell
            {
              buildInputs = buildInputs;
              env = env;
              shellHook = ''
                unset PYTHONPATH
              '';
            };

          devShells.cuda = pkgs.mkShell
            {
              buildInputs = buildInputs ++ (with pkgs;
                [
                  cudatoolkit
                  cudaPackages.cudnn
                ]);
              env = env;
              shellHook =
                let
                  drvPath = lib.makeLibraryPath (pkgs.pythonManylinuxPackages.manylinux1 ++ [
                    "/usr/lib/wsl" # WSL 2 stub   – keep first in order!
                    "/run/opengl-driver" # NixOS userspace driver path
                    pkgs.cudatoolkit
                    pkgs.cudaPackages.cudnn
                  ]);
                in
                ''
                  unset PYTHONPATH
                  export CUDA_PATH=${pkgs.cudatoolkit}
                  export LD_LIBRARY_PATH='${drvPath}:$LD_LIBRARY_PATH'
                '';
            };
        }
        );
}
