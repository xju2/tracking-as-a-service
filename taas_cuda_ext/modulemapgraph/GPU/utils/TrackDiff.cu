/***********************************************************************
 * Module Map Graph library
 * Track reconstruction for HEP
 * copyright © 2025,2026 COLLARD Christophe
 * Laboratoire des 2 Infinis - Toulouse - France (L2IT)
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 **********************************************************************/

#include <time.h>
#include <iostream>
#include <string>
#include <boost/program_options.hpp>
#include <MMG/common_options>
#include <MMG/parameters>
#include <MMG/colors>
#include <MMG/CUDA_edge_layer_connector>
#include <cuda_profiler_api.h>
#include <MMG/CUDA_vector_tools>
#include <MMG/CUDA_error_check>

//---------------------------------------------------------------------------------------------------------------------
int main (int argc, char* argv[]){
//---------------------------------------------------------------------------------------------------------------------
  int max_layers = 30;
  int max_tracks = 4000;
  int min_hits_per_track = 5;
  std::string filename = "track_350200_1.csv";
  // CC and Walk tracks
  std::string path = "/sps/atlas.new/w/wguerin/phd/workplace/run/gnn4itk/pub_note/data/v9/ATLAS-P2-RUN4-03-00-00/ttbar_eft_pu200/nominal/meanrms/relaxed_cuts/acorn_stages/track_building_stage/datasets/valset_tracks";
  CUDA_tracks<data_type1> CC_WALK_tracks (max_tracks, max_layers);
  CC_WALK_tracks.read (path, filename, min_hits_per_track);

  // Edge Layer Connector
  path = "/scratch_ssd/collard/OUTPUT_DIR_TRACKS_WARREN_13/valset_tracks";
  CUDA_tracks<data_type1> ELC_tracks (max_tracks, max_layers);
  ELC_tracks.read (path, filename, min_hits_per_track);

  // real edges
  path = "/scratch_ssd/data/gnn4itk/CTD25/v9/ATLAS-P2-RUN4-03-00-00/ttbar_eft_pu200/nominal/meanrms/relaxed_cuts/acorn_stages/edge_classifier_stage/dump_pyg2csv";
  filename = "000000001";
  CUDA_tracks<data_type1> real_tracks (2*max_tracks, 2*max_layers);
  real_tracks.read_real_tracks (path, filename, min_hits_per_track);


//  diff (ELC_tracks, CC_WALK_tracks, 1024);
  diff3 (ELC_tracks, CC_WALK_tracks, real_tracks, 1024);
}
