/***********************************************************************
 * Module Map Graph library
 * Track reconstruction for HEP
 * copyright © 2024,2025,2026 COLLARD Christophe
 * Laboratoire des 2 Infinis - Toulouse - France (L2IT)
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 **********************************************************************/

#include "MMG/tracks.cuh"
#include <limits>

//=====GPU functions for tracks========================================================================================

__global__ void print_tracks (int *nb_hits, int *hits, int max_tracks, int max_hits)
{
  printf ("\033[31mTRACKS %d \n", max_tracks);
  for (int i=0; i<max_tracks; i++)
    if (nb_hits[i]) {
      printf ("track # %d: nb hits: %d ", i, nb_hits[i]);
      for (int j=0; j<nb_hits[i]; j++)
        printf (" %d,", hits[i * max_hits + j]);
      printf ("\n");
    }
}

__global__ void print_track (int *nb_hits, int *hits, int track_nb, int max_hits)
{
  printf ("\033[31mTRACKS # %d, nb hits = %d \n", track_nb, nb_hits[track_nb]);
  if (nb_hits[track_nb]) {
    for (int j=0; j<nb_hits[track_nb]; j++)
      printf ("%d ", hits[track_nb * max_hits + j]);
    printf ("\n");
  }
}

__global__ void print_tracks (int i, int *positions,
                              bool *match_reconstructed, int *reconstructed_tracks, int *nb_reconstructed_tracks,
                              bool *match_real, int *real_tracks, int *nb_real_tracks,
                              int max_hits_per_track, int max_tracks)
{
  printf ("%d,%d\n", i, positions[i]-1);
//  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= max_tracks) return;

  printf ("tracks # %d \n", i);
  int p = i * max_hits_per_track;
  for (int k=0; k<nb_reconstructed_tracks[i]; k++)
    if (match_reconstructed[p+k]) printf ("\033[0;32m%d ", reconstructed_tracks[p+k]);
    else printf ("\033[0;31m%d ", reconstructed_tracks[p+k]);
  printf ("\033[0m \n");

  int position = positions[i]-1;
  if (position >= 0){
    printf ("linked tracks # %d \n", position);
    int q = position * max_hits_per_track;
    for (int k=0; k<nb_real_tracks[position]; k++)
      if (match_real[q+k]) printf ("\033[0;32m%d ", real_tracks[q+k]);
      else printf ("\033[0;31m%d ", real_tracks[q+k]);
    printf ("\033[0m \n");
  }
}

__global__ void print_tracks_error (int i, int *positions, bool *perfect,
                                    bool *match_reconstructed, int *reconstructed_tracks, int *nb_reconstructed_tracks,
                                    bool *match_real, int *real_tracks, int *nb_real_tracks,
                                    int max_hits_per_track, int max_tracks)
{
//  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= max_tracks) return;
  if (!nb_reconstructed_tracks[i]) return;
  if (perfect[i]) return;

  printf ("tracks # %d \n", i);
  int p = i * max_hits_per_track;
  for (int k=0; k<nb_reconstructed_tracks[i]; k++)
    if (match_reconstructed[p+k]) printf ("\033[0;32m%d ", reconstructed_tracks[p+k]);
    else printf ("\033[0;31m%d ", reconstructed_tracks[p+k]);
  printf ("\033[0m \n");

  int position = positions[i]-1;
  if (position >= 0){
    printf ("linked tracks # %d \n", position);
    int q = position * max_hits_per_track;
    for (int k=0; k<nb_real_tracks[position]; k++)
      if (match_real[q+k]) printf ("\033[0;32m%d ", real_tracks[q+k]);
      else printf ("\033[0;31m%d ", real_tracks[q+k]);
    printf ("\033[0m \n");
  }
}

__global__ void print_tracks_error (int i, int *positions12, int *positions13, bool *perfect,
                                    bool *match_track12, int *track1_hits, int *nb_track1_hits,
                                    bool *match_track21, int *track2_hits, int *nb_track2_hits,
                                    bool *match_track31, bool *match_track32, int *track3_hits, int *nb_track3_hits, uint64_t *track3_PID,
                                    int max_hits_per_track1, int max_tracks1,
                                    int max_hits_per_track2, int max_tracks2,
                                    int max_hits_per_track3, int max_tracks3){
//  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= max_tracks1) return;
  if (!nb_track1_hits[i]) return;
  if (perfect[i]) return;

  int position13 = positions13[i]-1;

  int p = i * max_hits_per_track1;
  int q = position13 * max_hits_per_track3;
  int nb_prefects_real = 0;
  for (int k=0; k<nb_track3_hits[position13]; k++)
    if (match_track31[q+k]) nb_prefects_real++;

  if (nb_track3_hits[position13] == nb_prefects_real && nb_track3_hits[position13] == nb_track1_hits[i])
    printf ("\033[0;32mtracks # %d \n", i);
  else
    printf ("\033[0;31mtracks # %d \n", i);

  for (int k=0; k<nb_track1_hits[i]; k++)
    if (match_track12[p+k]) printf ("\033[0;32m%d ", track1_hits[p+k]);
    else printf ("\033[0;31m%d ", track1_hits[p+k]);
  printf ("\033[0m \n");

  int position12 = positions12[i]-1;
  if (position12 >= 0){
    printf ("linked tracks # %d \n", position12);
    int r = position12 * max_hits_per_track1;
    for (int k=0; k<nb_track2_hits[position12]; k++)
      if (match_track21[r+k]) printf ("\033[0;32m%d ", track2_hits[r+k]);
      else printf ("\033[0;31m%d ", track2_hits[r+k]);
    printf ("\033[0m \n");
  }

  if (position13 >= 0){
    printf ("linked tracks PID # %ld \n", track3_PID[position13]);
    for (int k=0; k<nb_track3_hits[position13]; k++)
      if (match_track31[q+k] && match_track32[q+k]) printf ("\033[0;32m%d ", track3_hits[q+k]);
      else if (match_track31[q+k]) printf ("\033[0;34m%d ", track3_hits[q+k]);
      else if (match_track32[q+k]) printf ("\033[0;35m%d ", track3_hits[q+k]);
      else printf ("\033[0;31m%d ", track3_hits[q+k]);
    printf ("\033[0m \n");
  }

  printf ("---------------------------------------------------------------------------------------------------\n");
}

__global__ void print_track_reconstruction (int i, bool *match_reconstructed, bool *match_real, int *positions, bool *perfect,
                                            int *reconstructed_tracks, int *nb_reconstructed_tracks, int *real_tracks, int *nb_real_tracks,
                                            int max_hits_per_track, int min_hits_match, int max_tracks)
{
//  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= max_tracks) return;
  if (perfect[i]) return; // filter on perfect tracks

  if (!i) {
    printf ("track 0 \n");
    for (int k=0; k<nb_reconstructed_tracks[0]; k++)
      printf ("%d ", reconstructed_tracks[k]);
    printf ("\n");
  }
  if (!i) printf ("associated particle position = %d \n", positions[i]);

  if (positions[i] == -1) {
//    printf ("\033[0;35mNO MATCH \033[0m \n");
    return; // no match
  }

  int p = positions[i];
  int start = i * max_hits_per_track;
  if (real_tracks[nb_real_tracks[p]] != 112463 && reconstructed_tracks[start] != 112463) return; // print track segments for 

  for (int k=start; k<start+nb_reconstructed_tracks[i]; k++) {
    if (match_reconstructed[k]) printf ("\033[0;32m%d ", reconstructed_tracks[k]);
    else printf ("\033[0;31m%d ", reconstructed_tracks[k]);
  }
  printf ("\033[0m \n");

  for (int k=nb_real_tracks[p]; k<nb_real_tracks[p+1]; k++) {
    if (match_real[k]) printf ("\033[0;36m%d ", real_tracks[k]);
    else printf ("\033[0;35m%d ", real_tracks[k]);
  }
  printf ("\033[0m \n");
}

__global__ void print_statistics (int i, int *true_positive, int *false_positive, int *missed, int *positions, int *nb_reconstructed_tracks, int *nb_real_tracks, int max_tracks)
{
  if (i >= max_tracks) return;
  if (positions[i] == -1) return; // filter for tracks who are not matching at least n hits (walkthrough-min-hits) of a real track

  int p = positions[i];
  float TP = (float) true_positive[i] / (float) (nb_real_tracks[p+1] - nb_real_tracks[p]) * 100;
  printf ("True Positive = %.2f%%\n", TP);
  float FP = (float) false_positive[i] / (float) (nb_real_tracks[p+1] - nb_real_tracks[p]) * 100;
  printf ("False Positive = %.2f%%\n", FP);
  float FN = (float) missed[i] / (float) (nb_real_tracks[p+1] - nb_real_tracks[p]) * 100;
  printf ("False Positive = %.2f%%\n", FN);
}

//---------------------------------------------------------------------------------------------------------------------
template <class T>
__global__ void add_hits (int *nb_hits, int *hits, T *weights, int *last_vertex,
                          int layer_size,
                          int *source_vertex, int *target_vertex, T *layer_weights, int *tags,
                          int max_tracks, int max_hits){
//---------------------------------------------------------------------------------------------------------------------
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= layer_size) return;

  int tag = tags[i];
  if (!tag) return;

  tag--; // tag = 0 <-> deleted edge // tag = 1... tag value => stored data at tag-1
  int n = nb_hits[tag];
  int pos = tag * max_hits;
  if (!n) {
    hits[pos] = source_vertex[i];
    weights[tag] = layer_weights[i];
    n++;
  }

  hits[pos+n] = last_vertex[tag] = target_vertex[i];
  nb_hits[tag] = n+1;
}

// convert layers to tracks
//---------------------------------------------------------------------------------------------------------------------
template <class T>
__global__ void add_hits (int *nb_hits, int *hits, T *weights,
                          int *last_vertex, int layer_size,
                          int *source_vertex, int *target_vertex, T *layer_weights, int *tags,
                          const bool* mask, int max_tracks, int max_hits){
//---------------------------------------------------------------------------------------------------------------------
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= layer_size) return;

  int tag = tags[i];
  if (!tag) return;
  if (!mask[tag]) return;

  tag--; // tag = 0 <-> deleted edge // tag = 1... tag value => stored data at tag-1
  int n = nb_hits[tag];
  int pos = tag * max_hits;
  // store edge sources for layer 0
  if (!n) {
    hits[pos] = source_vertex[i];
    weights[tag] = layer_weights[i];
    n++;
  }

  // store edge targets for all layers
  hits[pos+n] = last_vertex[tag] = target_vertex[i];
  nb_hits[tag] = n+1;
}

//---------------------------------------------------------------------------------------------------------------------
__global__ void reverse_add_hits (int *nb_hits, int *hits,
                                  int layer_size, int *source_vertex, int *target_vertex, int *tags,
                                  int max_tracks, int max_hits){
//---------------------------------------------------------------------------------------------------------------------
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int tag = tags[i];
  if (i >= layer_size || !tag) return;

  int n = nb_hits[tag-1];
  if (!n) {
    hits[tag*max_hits-1] = target_vertex[i];
    nb_hits[tag-1]++;
  }
  hits[tag*max_hits-nb_hits[tag-1]-1] = source_vertex[i];
  nb_hits[tag-1]++;
}

//---------------------------------------------------------------------------------------------------------------------
__global__ void filter_tracks (bool *mask, int *nb_hits_per_track, int nb_min_hits, int nb_tracks){
//---------------------------------------------------------------------------------------------------------------------
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nb_tracks) return;

  mask[i] = (nb_hits_per_track[i] >= nb_min_hits);
}

//---------------------------------------------------------------------------------------------------------------------
template <class T>
__global__ void filter_tracks (bool *mask, int *last_vertex, int *nb_hits_per_track, T *weights, int nb_tracks)
//---------------------------------------------------------------------------------------------------------------------
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= nb_tracks || j >= nb_tracks || j<=i) return;
  if (!mask[i] || !mask[j]) return;
  if (last_vertex[i] != last_vertex[j]) return;

  int delta_nb = nb_hits_per_track[i] - nb_hits_per_track[j];

//  if (last_vertex[i] == 332765)
//    printf ("weights %f vs %f  and size %d vs %d and weights*nb_hits %f vs %f \n", weights[i], weights[j], nb_hits_per_track[i], nb_hits_per_track[j], weights[i]*nb_hits_per_track[i], weights[j]*nb_hits_per_track[j]);
//  int k = nb_hits_per_track[i] < nb_hits_per_track[j] ? i : j;
  int k = weights[i] * nb_hits_per_track[i] < weights[j] * nb_hits_per_track[j] ? i : j;
  if (delta_nb >= 10) k = j;
  if (delta_nb <= -10) k = i;

  mask[k] = 0;
//  if (last_vertex[i] == 332765)
//    printf ("kill chain %d in %d vs %d \n", nb_hits_per_track[k], nb_hits_per_track[i], nb_hits_per_track[j]);

//    printf ("weights %f / %f vs normalized weights %f / %f \n", weights[i], weights[j], weights[j]*nb_hits_per_track[j]);

//  int D_nb_hits = nb_hits_per_track[i] - nb_hits_per_track[j];
//  T D_weight = pow (0.5, D_nb_hits);

//  int k = nb_hits_per_track[i] * weights[i] < nb_hits_per_track[j] * weights[j] * D_weight ? i : j;
//  mask[k] = 0;
}

//---------------------------------------------------------------------------------------------------------------------------------------------------
__global__ void associate_reconstructed_real_tracks (int *positions,
                                                     int *reconstructed_tracks, int *nb_reconstructed_tracks, int * real_tracks, int *nb_real_tracks,
                                                     int max_hits_per_track, int min_hits_match, int max_tracks, int max_real_tracks){
//---------------------------------------------------------------------------------------------------------------------------------------------------
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= max_tracks) return;
  if (j >= max_real_tracks) return;

  int start = i * max_hits_per_track;
  int match = 0;
  if (i<3 && !j) printf ("max tracks = %d max real tracks = %d max hits per track = %d start = %d \n", max_tracks, max_real_tracks, max_hits_per_track, start);

  for (int k=start; k<start+nb_reconstructed_tracks[i]-1; k++)
    for (int l=nb_real_tracks[j]; l<nb_real_tracks[j+1]-1; l++)
      if (reconstructed_tracks[k] == real_tracks[l]  &&  reconstructed_tracks[k+1] == real_tracks[l+1])
        match++;


//  if (positions[i]) printf ("track already exists !!!!!!! \n");
  if (match >= min_hits_match) positions[i] = j;

/*
  if (match >= min_hits_match)
    if (i==2 || i==7) {
      printf ("i = %d, j = %d, positions[i] = %d \n", i, j, positions[i]);
      printf ("match for i = %d j = %d, start = %d, reconstructed_track = %d \n", i, j, start, reconstructed_tracks[start]);
      for (int k=start; k<start+nb_reconstructed_tracks[i]-1; k++)
        for (int l=nb_real_tracks[j]; l<nb_real_tracks[j+1]-1; l++)
          if (reconstructed_tracks[k] == real_tracks[l]  &&  reconstructed_tracks[k+1] == real_tracks[l+1])
            printf ("i = %d, j = %d, nb_real_tracks[j] = %d, l = %d, reconstructed = %d vs real = %d \n", i, j, nb_real_tracks[j], l, reconstructed_tracks[k], real_tracks[l]);
    }
*/
}

//------------------------------------------------------------------------------------------------------------------------------
__global__ void compare_tracks (bool *match_reconstructed, bool *match_real, int *positions,
                                int *reconstructed_tracks, int *nb_reconstructed_tracks, int * real_tracks, int *nb_real_tracks,
                                int max_hits_per_track, int min_hits_match, int max_tracks, int max_real_tracks){
//------------------------------------------------------------------------------------------------------------------------------
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= max_tracks) return;

  int start = i * max_hits_per_track;
  int p = positions[i];
  if (p == -1) return; // no correlation between real and reconstructed tracks

  for (int k=start; k<start+nb_reconstructed_tracks[i]-1; k++)
    for (int l=nb_real_tracks[p]; l<nb_real_tracks[p+1]-1; l++)
      if (reconstructed_tracks[k] == real_tracks[l]  &&  reconstructed_tracks[k+1] == real_tracks[l+1]) {
        match_reconstructed[k] = match_real[l] = true;
        match_reconstructed[k+1] = match_real[l+1] = true;
      }

/*
  if (real_tracks[nb_real_tracks[p]] == 112463) {
    printf ("i = %d, p = %d, nb_real_tracks = %d, real_tracks = %d\n", i, p, nb_real_tracks[p], real_tracks[nb_real_tracks[p]]);
    printf ("start = %d, positions = %d \n", start, positions[i]);
    printf ("\033[0;31mreconstructed vs real track \033[0;30m for track ID = %d \n", i+1);
    for (int k=start; k<start+nb_reconstructed_tracks[i]; k++)
      printf ("\033[0;30m%d \033[0;33m%d \033[0;32m%d ", i+1, k-start, reconstructed_tracks[k]);
    printf ("\n");
//    for (int k=start; k<start+nb_reconstructed_tracks[i]; k++)
//      printf ("\033[0;32m%d ", match_reconstructed[k]);
//    printf ("\n");
//    printf ("nb elements = %d \n", nb_real_tracks[p+1] - nb_real_tracks[p]);
    for (int l=nb_real_tracks[p]; l<nb_real_tracks[p+1]; l++)
      printf ("\033[0;30m%d \033[0;33m%d \033[0;34m%d ", i+1, l-nb_real_tracks[p], real_tracks[l]);
    printf ("\n");
  }
*/
}

//----------------------------------------------------------------------------------------------------------------------------------------------
__global__ void evaluate_track_reconstruction (int *true_positive, int *false_positive, int *missed, bool *perfect, bool *partial,
                                               bool *match_reconstructed, bool *match_real, int *positions,
                                               int *reconstructed_tracks, int *nb_reconstructed_tracks, int *real_tracks, int *nb_real_tracks,
                                               int max_hits_per_track, int max_tracks){
//----------------------------------------------------------------------------------------------------------------------------------------------
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= max_tracks) return;

  // no match
  if (positions[i] == -1) {
    true_positive[i] = 0;
    false_positive[i] = nb_reconstructed_tracks[i];
    missed[i] = 0;
    return;
  }

  int p = positions[i];
  int start = i * max_hits_per_track;
  int TP = 0;
  int TN = nb_real_tracks[p+1] - nb_real_tracks[p];

  for (int k=start; k<start+nb_reconstructed_tracks[i]; k++)
    TP += match_reconstructed[k];

  for (int k=nb_real_tracks[p]; k<nb_real_tracks[p+1]; k++)
    TN -= match_real[k];

  true_positive[i] = TP;
  false_positive[i] = nb_reconstructed_tracks[i] - TP;
  missed[i] = TN;

  perfect[i] = (nb_reconstructed_tracks[i] == nb_real_tracks[p+1] - nb_real_tracks[p]);
  partial[i] = !perfect[i] && (nb_reconstructed_tracks[i] >= 0.5 * (nb_real_tracks[p+1] - nb_real_tracks[p]));
  if (real_tracks[nb_real_tracks[p+1]-1] == 290060) {
    printf ("i = %d, nb reals tracks = %d \n", i, nb_real_tracks[p+1] - nb_real_tracks[p]);
    printf ("i = %d, nb reconstructed = %d \n", i, nb_reconstructed_tracks[i]);
    printf ("i = %d, TP = %d, TN = %d \n", i, TP, TN);
  }
}

/*
//------------------------------------------------------------------------------------------------------------------------------
__global__ void compare_tracks (bool *match_reconstructed, bool *match_real, int *positions,
                                int *reconstructed_tracks, int *nb_reconstructed_tracks, int * real_tracks, int *nb_real_tracks,
                                int max_hits_per_track, int min_hits_match, int max_tracks, int max_real_tracks)
//------------------------------------------------------------------------------------------------------------------------------
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (!i && !j) printf ("max tracks = %d / max_real_tracks = %d \n", max_tracks, max_real_tracks);
  if (!i && j==12221) printf ("compare edges\n");
  if (i >= max_tracks) return;
  if (j >= max_real_tracks) return;

  int start = i * max_hits_per_track;
  int match = 0;

  for (int k=start; k<start+nb_reconstructed_tracks[i]-1; k++)
    for (int l=nb_real_tracks[j]; l<nb_real_tracks[j+1]-1; l++)
      if (reconstructed_tracks[k] == real_tracks[l]  &&  reconstructed_tracks[k+1] == real_tracks[l+1]) {
        match_reconstructed[k] = match_real[l] = true;
        match_reconstructed[k+1] = match_real[l+1] = true;
        match++;
      }
  if (!i && j==12221) {
    printf ("track 0 & 12222 FOUND \n");
    for (int k=0; k<nb_reconstructed_tracks[0]; k++)
      printf ("%d ", reconstructed_tracks[k]);
    printf ("\n");
  }
  if (!i && j==12221) printf ("edges matched = %d \n", match);
  if (match >= min_hits_match) positions[i] = j;
}
*/

//----------------------------------------------------------------------------------------------------------------------------------------------
__global__ void link_tracks (int *link, bool *link_tracks1, bool *link_tracks2, bool *perfect,
                             int *tracks1_nb_hits, int *tracks1_hits, int tracks1_max_track, int tracks1_max_hits_per_track,
                             int *tracks2_nb_hits, int *tracks2_hits, int tracks2_max_track, int tracks2_max_hits_per_track){
//----------------------------------------------------------------------------------------------------------------------------------------------
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= tracks1_max_track || j >= tracks2_max_track) return;
  if (!tracks1_nb_hits[i] || !tracks2_nb_hits[j]) return;

  int nb_linked_hits = 0;

  int p = i * tracks1_max_hits_per_track;
  int q = j * tracks2_max_hits_per_track;

  for (int k=0; k<tracks1_nb_hits[i]; k++)
    for (int l=0; l<tracks2_nb_hits[j]; l++)
      if (tracks1_hits[p+k] == tracks2_hits[q+l]){
          link_tracks1[p+k] = link_tracks2[q+l] = true;
        nb_linked_hits++;
      }

  if (nb_linked_hits > 3)
    link[i] = j+1;

  if (tracks1_nb_hits[i] == tracks2_nb_hits[j]  &&  tracks1_nb_hits[i] == nb_linked_hits)
    perfect[i] = true;
}

//----------------------------------------------------------------------------------------------------------------------------------------------
__global__ void link_tracks_v2 (int *link, bool *link_tracks1, bool *link_tracks2, bool *perfect,
                                int *tracks1_nb_hits, int *tracks1_hits, int tracks1_max_track, int tracks1_max_hits_per_track,
                                int *tracks2_nb_hits, int *tracks2_hits, int tracks2_max_track, int tracks2_max_hits_per_track){
//----------------------------------------------------------------------------------------------------------------------------------------------
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= tracks1_max_track || j >= tracks2_max_track) return;
  if (!tracks1_nb_hits[i] || !tracks2_nb_hits[j]) return;
  int min_hits = 3;

  int nb_linked_hits = 0;

  int p = i * tracks1_max_hits_per_track;
  int q = j * tracks2_max_hits_per_track;

  for (int k=0; k<tracks1_nb_hits[i]; k++)
    for (int l=0; l<tracks2_nb_hits[j]; l++)
      if (tracks1_hits[p+k] == tracks2_hits[q+l]){
//          link_tracks1[p+k] = link_tracks2[q+l] = true;
        nb_linked_hits++;
      if (nb_linked_hits > min_hits) break;
      }

  if (nb_linked_hits > min_hits){
    for (int k=0; k<tracks1_nb_hits[i]; k++)
      for (int l=0; l<tracks2_nb_hits[j]; l++)
        if (tracks1_hits[p+k] == tracks2_hits[q+l])
          link_tracks1[p+k] = link_tracks2[q+l] = true;
    link[i] = j+1;
  }

  if (tracks1_nb_hits[i] == tracks2_nb_hits[j]  &&  tracks1_nb_hits[i] == nb_linked_hits)
    perfect[i] = true;
}

//----------------------------------------------------------------------------------------------------------------------------------------------
__global__ void link_tracks (int *link13, bool *link_tracks13, bool *link_tracks31, bool *link_tracks23, bool *link_tracks32,
                             int *link12, bool * perfect_true, bool *link_true,
                             int *tracks1_nb_hits, int *tracks1_hits, int tracks1_max_tracks, int tracks1_max_hits_per_track,
                             int *tracks2_nb_hits, int *tracks2_hits, int tracks2_max_tracks, int tracks2_max_hits_per_track,
                             int *tracks3_nb_hits, int *tracks3_hits, int tracks3_max_tracks, int tracks3_max_hits_per_track){
//----------------------------------------------------------------------------------------------------------------------------------------------
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= tracks1_max_tracks || j >= tracks3_max_tracks) return;
  if (!tracks1_nb_hits[i] || !tracks3_nb_hits[j]) return;

  int min_hits = 3;

  int p = i * tracks1_max_hits_per_track;
  int q = j * tracks3_max_hits_per_track;
  int r = (link12[i]-1) * tracks2_max_hits_per_track;

  int nb_linked_hits = 0;
  for (int k=0; k<tracks1_nb_hits[i]; k++)
    for (int l=0; l<tracks3_nb_hits[j]; l++)
      if (tracks1_hits[p+k] == tracks3_hits[q+l]){
//          link_tracks1[p+k] = link_tracks2[q+l] = true;
        nb_linked_hits++;
//      if (nb_linked_hits > min_hits) break;
      }

  if (nb_linked_hits == tracks3_nb_hits[j])
    perfect_true[i] = true;
  else if (nb_linked_hits >= 7)
    link_true[i] = true;

  if (nb_linked_hits > min_hits){
    for (int l=0; l<tracks3_nb_hits[j]; l++){
      for (int k=0; k<tracks1_nb_hits[i]; k++)
        if (tracks1_hits[p+k] == tracks3_hits[q+l])
          link_tracks13[p+k] = link_tracks31[q+l] = true;
      for (int k=0; k<tracks2_nb_hits[link12[i]-1]; k++)
        if (tracks2_hits[r+k] == tracks3_hits[q+l])
          link_tracks23[r+k] = link_tracks32[q+l] = true;
    }
    link13[i] = j+1;
  }
}


//=====CPU functions for tracks========================================================================================

//---------------------------------------------------------------------------------------------------------------------
template <class T>
CUDA_tracks<T>::CUDA_tracks (int max_tracks, int max_layers){
//---------------------------------------------------------------------------------------------------------------------
  _max_hits_per_track = max_layers+1; // nb hits = nb layers (source -> target) + 1
  _max_tracks = max_tracks;
  _size = _max_tracks * _max_hits_per_track;
  cudaMalloc (&_hits,  _size * sizeof(int));
  cudaMemset (_hits, 0, _size * sizeof(int));
  cudaMalloc (&_nb_hits, _max_tracks * sizeof(int));
  cudaMemset (_nb_hits, 0, _max_tracks * sizeof(int));
  cudaMalloc (&_weights, _max_tracks * sizeof (T));
  cudaMemset (_weights, 0, _max_tracks * sizeof(T));
  cudaMalloc (&_last_vertex, _max_tracks * sizeof(int));
  cudaMalloc (&_PID, _max_tracks * sizeof(uint64_t));
}

//---------------------------------------------------------------------------------------------------------------------
template <class T>
CUDA_tracks<T>::CUDA_tracks (tracks& cpu_tracks){
//---------------------------------------------------------------------------------------------------------------------
  _max_tracks = cpu_tracks.nb_tracks();
  _size = cpu_tracks.hits_by_track_sum()[_max_tracks];
  cudaMalloc (&_weights, _max_tracks * sizeof (T));
  cudaMalloc (&_last_vertex, _max_tracks * sizeof(int));

  // cp nb hits by track to gpu
  cudaMalloc (&_nb_hits, _max_tracks * sizeof(int));
  cudaMemcpy (_nb_hits, cpu_tracks.nb_hits_by_track().data(), _max_tracks*sizeof(int), cudaMemcpyHostToDevice);

  // cp hits to GPU
  cudaMalloc (&_hits, _size * sizeof(int));
  cudaMemset (_hits, 0, _size * sizeof(int));

  cudaMalloc (&_PID, _max_tracks * sizeof(uint64_t));

  for (int i=0; i<_max_tracks; i++)
    cudaMemcpy (_hits+cpu_tracks.hits_by_track_sum()[i], cpu_tracks[i].data(), cpu_tracks[i].size() * sizeof(int), cudaMemcpyHostToDevice);

  for (int i=0; i<3; i++)
    for (int j=0; j<cpu_tracks[i].size(); j++)
      std::cout << green << cpu_tracks[i][j] << " ";
  std::cout << reset;
  print_vector<<<1,1>>> (_hits, cpu_tracks.hits_by_track_sum()[1]);
  print_vector<<<1,1>>> (_hits, cpu_tracks.hits_by_track_sum()[1], cpu_tracks.hits_by_track_sum()[2]);
  print_vector<<<1,1>>> (_hits, cpu_tracks.hits_by_track_sum()[2], cpu_tracks.hits_by_track_sum()[3]);
  print_vector<<<1,1>>> (_nb_hits, 4);
}

//---------------------------------------------------------------------------------------------------------------------
template <class T>
CUDA_tracks<T>::~CUDA_tracks (){
//---------------------------------------------------------------------------------------------------------------------
  cudaFree (_PID);
  cudaFree (_last_vertex);
  cudaFree (_weights);
  cudaFree (_hits);
  cudaFree (_nb_hits);
}

//---------------------------------------------------------------------------------------------------------------------
template <class T>
void CUDA_tracks<T>::add (CUDA_edges<T>& layer, int blocks){
//---------------------------------------------------------------------------------------------------------------------
  dim3 grid_dim ((layer.nb_edges() + blocks - 1) / blocks);
  add_hits<<<grid_dim,blocks>>> (_nb_hits, _hits, _weights, _last_vertex,
                                layer.nb_edges(), layer.source(), layer.target(), layer.weight(), layer.tags(),
                                _max_tracks, _max_hits_per_track);
}

//---------------------------------------------------------------------------------------------------------------------
template <class T>
void CUDA_tracks<T>::add (CUDA_edges<T>& layer, CUDA_mask& mask, int blocks){
//---------------------------------------------------------------------------------------------------------------------
  dim3 grid_dim ((layer.nb_edges() + blocks - 1) / blocks);
  add_hits<<<grid_dim,blocks>>> (_nb_hits, _hits, _weights, _last_vertex,
                                layer.nb_edges(), layer.source(), layer.target(), layer.weight(), layer.tags(),
                                mask.cuda_mask(), _max_tracks, _max_hits_per_track);
}

//---------------------------------------------------------------------------------------------------------------------
template <class T>
void CUDA_tracks<T>::reverse_add (CUDA_edges<T>& layer, int blocks){
//---------------------------------------------------------------------------------------------------------------------
  dim3 grid_dim ((layer.nb_edges() + blocks - 1) / blocks);
  reverse_add_hits<<<grid_dim,blocks>>> (_nb_hits, _hits,
                                         layer.nb_edges(), layer.source(), layer.target(), layer.tags(),
                                         _max_tracks, _max_hits_per_track);
}

//---------------------------------------------------------------------------------------------------------------------
template <class T>
void CUDA_tracks<T>::filter (int min_hits, int blocks){
//---------------------------------------------------------------------------------------------------------------------
  if (!_max_tracks) return;

  dim3 grid_dim ((_max_tracks + blocks - 1) / blocks);
  CUDA_mask tracks_filter (_max_tracks);
  filter_tracks<<<grid_dim,blocks>>> (tracks_filter.cuda_mask(), _nb_hits, min_hits, _max_tracks);

  int blocks_2D = sqrt (blocks);
  dim3 block_dim_2D (blocks_2D, blocks_2D);
  int track_size = (_max_tracks + blocks_2D - 1) / blocks_2D;
  dim3 grid_dim_2D (track_size, track_size);
  filter_tracks<<<grid_dim_2D,block_dim_2D>>> (tracks_filter.cuda_mask(), _last_vertex, _nb_hits, _weights, _max_tracks);

  int *initial_positions;
  cudaMalloc (&initial_positions, _max_tracks * sizeof(int));
  init_vector<<<grid_dim,blocks>>> (initial_positions, _max_tracks);

  _max_tracks = tracks_filter.scan (blocks);
  if (!_max_tracks) {
    cudaFree (initial_positions);
    cudaFree (_hits);
    cudaFree (_nb_hits);
    cudaFree (_last_vertex);
    cudaFree (_weights);
    cudaFree (_PID);
    _size = 0;
    _hits = nullptr;
    _nb_hits = nullptr;
    _last_vertex = nullptr;
    _weights = nullptr;
    _PID = nullptr;
    cudaDeviceSynchronize();
    GPU_ERROR_CHECK (cudaGetLastError());
    return;
  }
  _size = _max_tracks * _max_hits_per_track;

  compact_stream (&initial_positions, tracks_filter, _max_tracks, blocks);
  compact_stream (&_nb_hits, tracks_filter, _max_tracks, blocks);
  compact_stream (&_last_vertex, tracks_filter, _max_tracks, blocks); // delete after debug
  compact_stream (&_weights, tracks_filter, _max_tracks, blocks); // delete after debug

  grid_dim = ((_max_tracks + blocks - 1) / blocks);
  int *reduced_hits;
  cudaMalloc (&reduced_hits, _size * sizeof(int));
  compact_stream<<<grid_dim,blocks>>> (reduced_hits, _hits, _max_hits_per_track, initial_positions, _max_tracks);
  cudaFree (_hits);
  _hits = reduced_hits;
  reduced_hits=nullptr;

  cudaDeviceSynchronize();
  GPU_ERROR_CHECK (cudaGetLastError());
}

//---------------------------------------------------------------------------------------------------------------------
template <class Tf>
void diff (const CUDA_tracks<Tf>& tracks1, const CUDA_tracks<Tf>& tracks2, int blocks){
//---------------------------------------------------------------------------------------------------------------------
  assert (tracks1._max_hits_per_track == tracks2._max_hits_per_track);
  assert (tracks1._max_tracks == tracks2._max_tracks);

  int size = tracks1._max_tracks;
  int *links;
  bool *link_tracks1, *link_tracks2;
  cudaMalloc (&links, tracks1._max_tracks * sizeof(int));
  cudaMalloc (&link_tracks1, tracks1._size * sizeof(bool));
  cudaMalloc (&link_tracks2, tracks2._size * sizeof(bool));
  cudaMemsetAsync (links, 0, tracks1._max_tracks * sizeof(int));
  cudaMemsetAsync (link_tracks1, 0, tracks1._size * sizeof(bool));
  cudaMemsetAsync (link_tracks2, 0, tracks2._size * sizeof(bool));
  CUDA_mask perfect (tracks1._max_tracks);

  dim3 blocks2D (sqrt(blocks), sqrt(blocks));
  int trk1_dim = (tracks1._max_tracks + blocks2D.x - 1) / blocks2D.x;
  int trk2_dim = (tracks2._max_tracks + blocks2D.y - 1) / blocks2D.y;
  dim3 grid_dim2D (trk1_dim, trk2_dim);
  link_tracks<<<grid_dim2D,blocks2D>>> (links, link_tracks1, link_tracks2, perfect.cuda_mask(),
                                    tracks1._nb_hits, tracks1._hits, tracks1._max_tracks, tracks1._max_hits_per_track,
                                    tracks2._nb_hits, tracks2._hits, tracks2._max_tracks, tracks2._max_hits_per_track);

  int nb_perfects = perfect.scan(blocks);
  std::cout << "nb perfects = " << nb_perfects << reset;

//  dim3 grid ((tracks1._max_tracks + blocks - 1) / blocks);
  for (int i=0; i<tracks1._max_tracks; i++)
//  for (int i=0; tracks1._nb_hits[i]; i++)
    print_tracks_error<<<1,1>>> (i, links, perfect.cuda_mask(),
                                 link_tracks1, tracks1._hits, tracks1._nb_hits,
                                 link_tracks2, tracks2._hits, tracks2._nb_hits,
                                 tracks1._max_hits_per_track, tracks1._max_tracks);

/*
  int i = 0;
  print_tracks<<<1,1>>>(i, links,
                        link_tracks1, tracks1._hits, tracks1._nb_hits,
                        link_tracks2, tracks2._hits, tracks2._nb_hits,
                        tracks1._max_hits_per_track, tracks1._max_tracks);
*/
}

//---------------------------------------------------------------------------------------------------------------------
template <class Tf>
void diff3(const CUDA_tracks<Tf>& tracks1, const CUDA_tracks<Tf>& tracks2, const CUDA_tracks<Tf>& tracks3, int blocks){
//---------------------------------------------------------------------------------------------------------------------
  assert (tracks1._max_hits_per_track == tracks2._max_hits_per_track);
  assert (tracks1._max_tracks == tracks2._max_tracks);

  int size = tracks1._max_tracks;
  int *links12;
  bool *link_tracks12, *link_tracks21;
  cudaMalloc (&links12, tracks1._max_tracks * sizeof(int));
  cudaMalloc (&link_tracks12, tracks1._size * sizeof(bool));
  cudaMalloc (&link_tracks21, tracks2._size * sizeof(bool));
//  cudaMalloc (&link_tracks3, tracks3._size * sizeof(bool));
  cudaMemsetAsync (links12, 0, tracks1._max_tracks * sizeof(int));
  cudaMemsetAsync (link_tracks12, 0, tracks1._size * sizeof(bool));
  cudaMemsetAsync (link_tracks21, 0, tracks2._size * sizeof(bool));
//  cudaMemsetAsync (link_tracks3, 0, tracks3._size * sizeof(bool));
  CUDA_mask perfect (tracks1._max_tracks), perfect_true (tracks1._max_tracks), link_true (tracks1._max_tracks);

  dim3 blocks2D (sqrt(blocks), sqrt(blocks));
  int trk1_dim = (tracks1._max_tracks + blocks2D.x - 1) / blocks2D.x;
  int trk2_dim = (tracks2._max_tracks + blocks2D.y - 1) / blocks2D.y;
  int trk3_dim = (tracks3._max_tracks + blocks2D.y - 1) / blocks2D.y;
  dim3 grid_dim2D (trk1_dim, trk2_dim);
  link_tracks_v2<<<grid_dim2D,blocks2D>>> (links12, link_tracks12, link_tracks21, perfect.cuda_mask(),
                                    tracks1._nb_hits, tracks1._hits, tracks1._max_tracks, tracks1._max_hits_per_track,
                                    tracks2._nb_hits, tracks2._hits, tracks2._max_tracks, tracks2._max_hits_per_track);

  int nb_perfects = perfect.scan(blocks);
  std::cout << magenta << "nb perfects = " << nb_perfects << reset;

  std::cout << "real tracks size = " << tracks3._max_tracks << reset;
///*
  int *links13;
  cudaMalloc (&links13, tracks1._max_tracks * sizeof(int));
  bool *link_tracks13, *link_tracks31, *link_tracks23, *link_tracks32;
  cudaMalloc (&link_tracks13, tracks1._size * sizeof(bool));
  cudaMalloc (&link_tracks23, tracks2._size * sizeof(bool));
  cudaMalloc (&link_tracks31, tracks3._size * sizeof(bool));
  cudaMalloc (&link_tracks32, tracks3._size * sizeof(bool));
  cudaMemsetAsync (link_tracks13, 0, tracks1._size * sizeof(bool));
  cudaMemsetAsync (link_tracks31, 0, tracks2._size * sizeof(bool));
  cudaMemsetAsync (link_tracks23, 0, tracks1._size * sizeof(bool));
  cudaMemsetAsync (link_tracks32, 0, tracks2._size * sizeof(bool));
  dim3 grid_dim_2D (trk1_dim, trk3_dim);
std::cout << "grid size = " << trk1_dim << " " << trk3_dim << reset;
std::cout << "block size = " << blocks2D.x << "," << blocks2D.y << reset;
std::cout << "nb elements = " << grid_dim2D.y * blocks2D.y << reset;
//  print_track<<<1,1>>> (tracks3._nb_hits, tracks3._hits, 0, tracks3._max_hits_per_track);
//  print_track<<<1,1>>> (tracks3._nb_hits, tracks3._hits, 1, tracks3._max_hits_per_track);
//  print_track<<<1,1>>> (tracks3._nb_hits, tracks3._hits, 2, tracks3._max_hits_per_track);

  link_tracks<<<grid_dim_2D,blocks2D>>> (links13, link_tracks13, link_tracks31, link_tracks23, link_tracks32,
                                    links12, perfect_true.cuda_mask(), link_true.cuda_mask(),
                                    tracks1._nb_hits, tracks1._hits, tracks1._max_tracks, tracks1._max_hits_per_track,
                                    tracks2._nb_hits, tracks2._hits, tracks2._max_tracks, tracks2._max_hits_per_track,
                                    tracks3._nb_hits, tracks3._hits, tracks3._max_tracks, tracks3._max_hits_per_track);

  int nb_perfects_true = perfect_true.scan(blocks);
  std::cout << blue << "nb perfects true = " << nb_perfects_true << reset;

  int nb_true = link_true.scan(blocks);
  std::cout << blue << "nb link true (not perfect) = " << nb_true << reset;

//*/
//  dim3 grid ((tracks1._max_tracks + blocks - 1) / blocks);
//  for (int i=0; i<tracks1._max_tracks; i++)
  for (int i=0; i<600; i++)
    print_tracks_error<<<1,1>>> (i, links12, links13, perfect.cuda_mask(),
                                 link_tracks12, tracks1._hits, tracks1._nb_hits,
                                 link_tracks21, tracks2._hits, tracks2._nb_hits,
                                 link_tracks31, link_tracks32, tracks3._hits, tracks3._nb_hits, tracks3._PID,
                                 tracks1._max_hits_per_track, tracks1._max_tracks,
                                 tracks2._max_hits_per_track, tracks2._max_tracks,
                                 tracks3._max_hits_per_track, tracks3._max_tracks);

/*
  int i = 0;
  print_tracks<<<1,1>>>(i, links,
                        link_tracks1, tracks1._hits, tracks1._nb_hits,
                        link_tracks2, tracks2._hits, tracks2._nb_hits,
                        tracks1._max_hits_per_track, tracks1._max_tracks);
*/
}

//---------------------------------------------------------------------------------------------------------------------
template <class T>
void CUDA_tracks<T>::read (const std::string& path, const std::string& filename, int min_hits){
//---------------------------------------------------------------------------------------------------------------------
  std::vector<int> nb_hits (_max_tracks, 0);
  std::vector<int> hits (_size, 0);
  std::vector<int> last_vertex (_max_tracks, 0);

  std::string full_filename = path + "/" + filename;
  // get the nb of tracks in file <filename>
  std::string log_tracks_filenames = "log/" + filename + ".log";
  std::string cmd_tracks = "rm " + log_tracks_filenames;
  system (cmd_tracks.c_str());
  cmd_tracks = "wc -l " + full_filename + " >> " + log_tracks_filenames;
  system (cmd_tracks.c_str());

  std::ifstream log_file (log_tracks_filenames);
  if (log_file.fail())
    throw std::invalid_argument("Cannot open file " + log_tracks_filenames);

  int max_tracks;
  log_file >> max_tracks;
  log_file.close();
  std::cout << "nb tracks in file = " << max_tracks << reset;

  // read all the tracks as a text file
  std::ostringstream ss;
  std::ifstream file (full_filename);
  if (file.fail()) throw std::invalid_argument ("Cannot open file " + full_filename);

  // block reading
  ss << file.rdbuf();
  file.close ();

  std::istringstream sstream (ss.str());
  std::string content, hitID;
  char delim = ',';

  int nb_tracks = 0;
  for (;getline (sstream, content);){   // read 1 line of text
    std::stringstream ssi (content);    // convert line to stream
    std::vector<int> track;
    for (int i=0; getline (ssi, hitID, delim); i++) // read line content
      track.push_back(std::stoi(hitID));            // store hitsID in track vector

    int n = track.size();
    if (n >= min_hits) {
      nb_hits[nb_tracks] = n;
      for (int i=0; i<n; i++){
        hits[nb_tracks*_max_hits_per_track+i] = track[i];
//        std::cout << track[i] << " ";
      }
      last_vertex[nb_tracks] = track[n-1];
      nb_tracks++;
      assert (nb_tracks <= _max_tracks);
//      std::cout << reset;
    }
  }

  std::cout << "nb tracks = " << nb_tracks << reset;
  cudaMemcpyAsync (_nb_hits, nb_hits.data(), _max_tracks*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpyAsync (_hits, hits.data(), _size*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpyAsync (_last_vertex, last_vertex.data(), _size*sizeof(int), cudaMemcpyHostToDevice);
}

//---------------------------------------------------------------------------------------------------------------------
template <class T>
void CUDA_tracks<T>::read_real_tracks (const std::string& path, const std::string& filename, int min_hits){
//---------------------------------------------------------------------------------------------------------------------
  constexpr uint64_t INVALID_PARTICLE_ID = std::numeric_limits<uint64_t>::max();

  // --------------------read hitID
  std::vector<int> ID;
  std::string full_filename = path + "/" + filename + "-hit.csv";
  std::ostringstream ss_hits;
  std::ifstream file_hits (full_filename);
  if (file_hits.fail()) throw std::invalid_argument ("Cannot open file " + full_filename);

  // block reading
  ss_hits << file_hits.rdbuf();
  file_hits.close ();

  std::istringstream sstream0 (ss_hits.str());
  std::string content, hitID;
  char delim = ',';

  // delete first line = comments
  getline (sstream0, content);

  for (;getline (sstream0, content);){  // read 1 line of text
    std::stringstream ssi (content);    // convert line to stream
    std::vector<int> track;
    getline (ssi, hitID, delim);        // read line content
    ID.push_back(std::stoi(hitID));     // store hitsID in track vector
  }

  // --------------------read edges

  full_filename = path + "/" + filename + "-edge.csv";

  // get the nb of edges in file <filename>
  std::string log_tracks_filenames = "log/" + filename + ".log";
  std::string cmd_tracks = "rm " + log_tracks_filenames;
  system (cmd_tracks.c_str());
  cmd_tracks = "wc -l " + full_filename + " >> " + log_tracks_filenames;
  system (cmd_tracks.c_str());

  std::ifstream log_file (log_tracks_filenames);
  if (log_file.fail())
    throw std::invalid_argument("Cannot open file " + log_tracks_filenames);

  int max_tracks;
  log_file >> max_tracks;
  log_file.close();
  std::cout << "nb edges in file = " << max_tracks << reset;

  // read all the tracks as a text file
  std::ostringstream ss;
  std::ifstream file (full_filename);
  if (file.fail()) throw std::invalid_argument ("Cannot open file " + full_filename);

  // block reading
  ss << file.rdbuf();
  file.close ();

  std::istringstream sstream (ss.str());
  // delete first line = comments
  getline (sstream, content);

  // store index names
  std::stringstream ssi (content);
  std::vector<std::string> column_id;
  std::string line;
  std::map<std::string,int> index;
  for (int i=0; getline(ssi, line, delim); i++)
    index[line] = i;

  std::multimap<uint64_t, std::vector<int>> real_tracks;
  std::string item;
  for (;getline (sstream, content);){   // read 1 line of text
    std::stringstream ssi (content);    // convert line to stream
    std::vector<std::string> columns;
    for (int i=0; getline (ssi, item, delim); i++) // read line content
      columns.push_back(item);            // store hitsID in track vector

    uint64_t pid;
    std::istringstream (columns[index.find("particle_id")->second]) >> pid;
    int edge_source, edge_target;
    std::istringstream (columns[index.find("edge_src")->second]) >> edge_source;
    std::istringstream (columns[index.find("edge_dst")->second]) >> edge_target;
    if (pid != INVALID_PARTICLE_ID){
      bool new_source=true, new_target=true, id=false;
      for (auto it = real_tracks.begin(); it != real_tracks.end(); ++it)
        if (it->first == pid){
          id=true;
          for (auto vec_it = it->second.begin(); vec_it != it->second.end(); ++vec_it){
            new_source *= (*vec_it != edge_source);
            new_target *= (*vec_it != edge_target);
          }
          if (new_source) (it->second).push_back (ID[edge_source]);
          if (new_target) (it->second).push_back (ID[edge_target]);
//          if (new_source) (it->second).push_back (edge_source);
//          if (new_target) (it->second).push_back (edge_target);


        }
      if (!id) real_tracks.insert ({pid, {edge_source, edge_target}});
    }
  }

  std::cout << "nb tracks = " << real_tracks.size() << reset;

  // count the nb of tracks and store them into cpu memory
  int nb_tracks = 0;
  std::vector<uint64_t> pid(_max_tracks, INVALID_PARTICLE_ID);
  std::vector<int> nb_hits(_max_tracks,0);
  std::vector<int> cpu_tracks(_size, 0);
  for (const std::pair<uint64_t, std::vector<int>> entry : real_tracks)
    if (entry.second.size() >= min_hits){
      nb_hits[nb_tracks] = entry.second.size();
      pid[nb_tracks] = entry.first;
      for (int i=0; i<entry.second.size(); i++)
        cpu_tracks[nb_tracks*_max_hits_per_track+i] = entry.second[i];
        // pid.push_back (entry.first);
      nb_tracks++;
    }

  std::cout << "nb tracks larger >= " << min_hits << " hits = " << nb_tracks << reset;

  // send tracks to GPU
  cudaMemcpyAsync (_nb_hits, nb_hits.data(), nb_hits.size()*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpyAsync (_hits, cpu_tracks.data(), _size*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpyAsync (_PID, pid.data(), _max_tracks*sizeof(uint64_t), cudaMemcpyHostToDevice);
//  print_track<<<1,1>>> (_nb_hits, _hits, 0, _max_hits_per_track);
//  print_track<<<1,1>>> (_nb_hits, _hits, 1, _max_hits_per_track);
//  print_track<<<1,1>>> (_nb_hits, _hits, 2, _max_hits_per_track);
}

//---------------------------------------------------------------------------------------------------------------------
template <class T>
void CUDA_tracks<T>::save (int min_hits, const std::string& filename){
//---------------------------------------------------------------------------------------------------------------------
  std::cout << "copy data GPU -> CPU" << std::endl;
  std::vector<int> nb_hits(_max_tracks), hits(_size);
  std::vector<T> weights (_max_tracks); // delete after debug
  std::vector<int> last_vertex (_max_tracks); // delete after debug

  cudaMemcpy (nb_hits.data(), _nb_hits, _max_tracks * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy (hits.data(), _hits, _size * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy (weights.data(), _weights, _max_tracks * sizeof(T), cudaMemcpyDeviceToHost); // delete after debug
  cudaMemcpy (last_vertex.data(), _last_vertex, _max_tracks * sizeof(int), cudaMemcpyDeviceToHost); // delete after debug

  std::cout << "removing file " << filename << std::endl;
  std::string cmd = "rm " + filename;
  std::cout << "cmd = " << cmd;
  system (cmd.c_str());

  std::cout << "saving data ...." << reset;
  std::ofstream outfile;
  outfile.open(filename.c_str(), std::ios_base::app);
  for (int i=0; i<_max_tracks; i++)
    if (nb_hits[i] >= min_hits) {
      for (int j=0; j<nb_hits[i]; j++) {
        int hitID = hits[i*_max_hits_per_track+j];
        outfile << hitID << " ";
      }
//      outfile << " | " << nb_hits[i]*weights[i] << " | " << last_vertex[i]; // delete after debug
      outfile << std::endl;
    }
  outfile.close();

  std::cout << "max tracks = " << _max_tracks << std::endl;
}

//---------------------------------------------------------------------------------------------------------------------
template <class T>
void CUDA_tracks<T>::save (int min_hits, const std::vector<int>& hitsID,  std::string& filename){
//---------------------------------------------------------------------------------------------------------------------
  std::cout << "copy data GPU -> CPU" << std::endl;
  std::vector<int> nb_hits(_max_tracks), hits(_size);
  std::vector<T> weights (_max_tracks); // delete after debug
  std::vector<int> last_vertex (_max_tracks); // delete after debug
  cudaMemcpy (nb_hits.data(), _nb_hits, _max_tracks * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy (hits.data(), _hits, _size * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy (weights.data(), _weights, _max_tracks * sizeof(T), cudaMemcpyDeviceToHost); // delete after debug
  cudaMemcpy (last_vertex.data(), _last_vertex, _max_tracks * sizeof(int), cudaMemcpyDeviceToHost); // delete after debug

  std::cout << "removing file " << filename << std::endl;
  std::string cmd = "rm " + filename;
  system (cmd.c_str());

  std::cout << "saving data ...." << reset;
  int maxID = hitsID.size();
  std::ofstream outfile;
  outfile.open(filename.c_str(), std::ios_base::app);
  for (int i=0; i<_max_tracks; i++)
    if (nb_hits[i] >= min_hits) {
      for (int j=0; j<nb_hits[i]; j++) {
        int indice = hits[i*_max_hits_per_track+j];
        assert (indice < maxID);
        outfile << hitsID[indice] << " ";
      }
      outfile << " | " << nb_hits[i]*weights[i] << " | " << last_vertex[i]; // delete after debug
      outfile << std::endl;
    }
  outfile.close();

  std::cout << "max tracks = " << _max_tracks << std::endl;
}

//--------------------------------------------------------------------------------------------------------------
template <class T>
void CUDA_tracks<T>::save (int min_hits, CUDA_mask& mask, std::string& filename){
//--------------------------------------------------------------------------------------------------------------
  std::cout << "copy data GPU -> CPU" << std::endl;
  std::vector<int> nb_hits(_max_tracks), hits(_size);
  bool *track_mask;
  assert (mask.size());
  track_mask = new bool [mask.size()];
  cudaMemcpy (nb_hits.data(), _nb_hits, _max_tracks * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy (hits.data(), _hits, _size * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy (track_mask, mask.cuda_mask(), _max_tracks * sizeof(bool), cudaMemcpyDeviceToHost);

  std::cout << "removing file " << filename << std::endl;
  std::string cmd = "rm " + filename;
  system (cmd.c_str());

  std::cout << "saving data ...." << reset;
  std::ofstream outfile;
  outfile.open(filename.c_str(), std::ios_base::app);
  for (int i=0; i<_max_tracks; i++)
    if (nb_hits[i] >= min_hits && track_mask[i]) {
      for (int j=0; j<nb_hits[i]; j++) {
        int hitID = hits[i*_max_hits_per_track+j];
        outfile << hitID << " ";
      }
      outfile << std::endl;
    }
  outfile.close();
  delete [] track_mask;

  std::cout << "max tracks = " << _max_tracks << std::endl;
}
