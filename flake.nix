{
  description = "nix declared development environment";

  inputs = {
    nixpkgs.url = "nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { nixpkgs, flake-utils, ... }:
    let
      mkshell = pkgs:
        with pkgs;
        mkShell rec {
          buildInputs = [
            poetry
            python39
            pre-commit
            pandoc
            git
            cacert
            stdenv
            pastel
            zlib
            nixFlakes
            dos2unix
            expat
            # cudnn_8_3_cudatoolkit_11_5
          ];

          # Required for building C extensions
          LD_LIBRARY_PATH =
            # "${stdenv.cc.cc.lib}/lib:${zlib}/lib:${expat}/lib:${cudnn_8_3_cudatoolkit_11_5.cudatoolkit.lib}/lib";
            "${stdenv.cc.cc.lib}/lib:${zlib}/lib:${expat}/lib";
          # Certificates for secure connections for e.g. pip downloads
          GIT_SSL_CAINFO = "${cacert}/etc/ssl/certs/ca-bundle.crt";
          SSL_CERT_FILE = "${cacert}/etc/ssl/certs/ca-bundle.crt";
          CURL_CA_BUNDLE = "${cacert}/etc/ssl/certs/ca-bundle.crt";
          # Required to fully use the python environments
          PYTHON39PATH = "${python39}/lib/python3.9/site-packages";
          # PYTHONPATH is overridden with contents from e.g. poetry */site-package.
          # We do not want them to be in PYTHONPATH.
          # Therefore, in ./.envrc PYTHONPATH is set to the _PYTHONPATH defined below
          # and also in shellHooks (direnv does not load shellHook exports, always).
          _PYTHONPATH = "${PYTHON39PATH}";

          envrc_contents = ''
            use flake
            export PYTHONPATH=$_PYTHONPATH
          '';

          shellHook = ''
            [[ -a .pre-commit-config.yaml ]] && \
              echo "Installing pre-commit hooks"; pre-commit install
            pastel paint -n green "
            Run poetry install to install environment from poetry.lock
            "
            export PYTHONPATH=$_PYTHONPATH
            [[ ! -a .envrc ]] && echo -n "$envrc_contents" > .envrc
          '';
        };
    in (flake-utils.lib.eachDefaultSystem (system:
      let
        unfree_nixpkgs = import nixpkgs {
          config = { allowUnfree = true; };
          inherit system;
        };
        # pkgs = unfree_nixpkgs.legacyPackages."${system}";
        # # Helper function to set allowUnfree for nixpkgs
        # setup_pkgs = { pkgs }:
        # import pkgs {
        # inherit system;
        # config = { allowUnfree = true; };
        # };
        # pkgs_unfree = setup_pkgs { inherit pkgs; };

      in { devShell = mkshell unfree_nixpkgs; }));
}
