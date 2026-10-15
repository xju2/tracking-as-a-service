#!/bin/bash
#***********************************************************************
#* Module Map Graph library
#* Track reconstruction for HEP
#* copyright © 2026 COLLARD Christophe
#* Laboratoire des 2 Infinis - Toulouse - France (L2IT)
#*
#* This Source Code Form is subject to the terms of the Mozilla Public
#* License, v. 2.0. If a copy of the MPL was not distributed with this
#* file, You can obtain one at https://mozilla.org/MPL/2.0/.
#***********************************************************************/

[[ -d install/bin ]] && rm install/bin
ln -sf $MMG_path/install/nvcc-clang++/bin $MMG_path/install/bin
[[ -d install/lib ]] && rm install/lib
ln -sf $MMG_path/install/nvcc-clang++/lib $MMG_path/install/lib
[[ -d install/lib64 ]] && rm install/lib64
ln -sf $MMG_path/install/nvcc-clang++/lib64 $MMG_path/install/lib64
[[ -d OUTPUT_DIR_GEN ]] && rm -Rf OUTPUT_DIR_GEN
mkdir OUTPUT_DIR_GEN
export ext="cu.exe"
PS3='please choose a program to execute: '
programs=("Graph Builder" "Graph Builder (threads)" "Graph Builder (slurm)" "Edge Layer Connector" "Track Diff" "Return to main menu")
select opt in "${programs[@]}"; do
  case $opt in
    "Graph Builder")
      source $MMG_path/scripts/GraphBuilder.sh
      ;;
    "Graph Builder (threads)")
      export ext=threads.$ext
      source $MMG_path/scripts/GraphBuilder.sh
      export ext=${ext#*.}
      ;;
    "Graph Builder (slurm)")
      source $MMG_path/GPU/scripts/GraphBuilder.slurm.sh
      ;;
    "Edge Layer Connector")
      source $MMG_path/GPU/scripts/EdgeLayerConnector.sh
      ;;
    "Track Diff")
      source $MMG_path/GPU/scripts/TrackDiff.sh
      ;;
    "Return to main menu")
      echo -e "\033[34mMain Menu\033[0m"
      break
      ;;
    *) echo -e "\033[31minvalid option\033[0m $REPLY"
      ;;
  esac
  REPLY=
done
