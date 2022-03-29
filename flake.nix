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
            python38
            pre-commit
            pandoc
            git
            cacert
            stdenv
            pastel
            zlib
            nixFlakes
          ];

          # Required for building C extensions
          LD_LIBRARY_PATH = "${stdenv.cc.cc.lib}/lib:${zlib}/lib";
          # Certificates for secure connections for e.g. pip downloads
          GIT_SSL_CAINFO = "${cacert}/etc/ssl/certs/ca-bundle.crt";
          SSL_CERT_FILE = "${cacert}/etc/ssl/certs/ca-bundle.crt";
          CURL_CA_BUNDLE = "${cacert}/etc/ssl/certs/ca-bundle.crt";
          # Required to fully use the python environments
          PYTHON38PATH = "${python38}/lib/python3.8/site-packages";
          # PYTHONPATH is overridden with contents from e.g. poetry */site-package.
          # We do not want them to be in PYTHONPATH.
          # Therefore, in ./.envrc PYTHONPATH is set to the _PYTHONPATH defined below
          # and also in shellHooks (direnv does not load shellHook exports, always).
          _PYTHONPATH = "${PYTHON38PATH}";

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
      let pkgs = nixpkgs.legacyPackages."${system}";
      in { devShell = mkshell pkgs; }));
}
