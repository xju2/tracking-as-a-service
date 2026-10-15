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

#include "MMG/edge_layer_connector.cuh"

template <class T>
__global__ void print (T* vector, int* mask, int size)
{
  for (int i=0; i<size; i++)
    printf ("%d ", vector[mask[i]]);
    printf ("\n");
}

template <class T>
__global__ void print (int *v1, int *v2, T *v3, int *pos, bool *mask, int size)
{
  for (int i=0; i<size; i++)
    if (mask[i])
      printf ("i = %d / source = %d target = %d weight = %f \n", i, v1[pos[i]], v2[pos[i]], v3[pos[i]]);
}

template <class T>
__global__ void print (T* vector, int size, char* txt="")
{
  printf ("print vector from edge_layer_connector.cu \n");
  for (int i=0; i<size; i++)
    printf ("\033[31m%d ", vector[i]);
    printf ("\n");
}

__global__ void print_layer (int *connected_edges, int size_ce, int *new_edges, int size_ne)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i) return;

  for (int i=0; i<size_ce; i++)
    printf ("\033[32m%d ", connected_edges[i]);
  for (int i=0; i<size_ne; i++)
    printf ("\033[31m%d ", new_edges[i]);

  printf ("\033[0;30m\n");
}

__global__ void print_layer_double (data_type1 *connected_edges, int size_ce, data_type1 *new_edges, int size_ne)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i) return;

  for (int i=0; i<size_ce; i++)
    printf ("\033[32m%f ", connected_edges[i]);
  for (int i=0; i<size_ne; i++)
    printf ("\033[31m%f ", new_edges[i]);

  printf ("\033[0;30m\n");
}

__global__ void print_hits_in_tracks (bool *hits, int nb_hits)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i) return;

  for (int k=0; k<nb_hits; k++)
    if (hits[k]) printf ("\033[0;34m%d ", k);

  printf ("\033[0;30m \n");
}

//---------------------------------------------------------------------------------------------------------------------
__global__ void print_last_edges (int *pos, bool *mask, int *source, int* target, int size){
//---------------------------------------------------------------------------------------------------------------------
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i) return;

  for (int k=0; k<size; k++) {
    int p = pos[k];
    printf ("edge: %d - %d tag = %d \n", source[p], target[p], mask[p]);
  }

  printf ("\n");
}

__global__ void null_kernel(){
}

//=====GPU functions for edge layer connector==========================================================================


//---------------------------------------------------------------------------------------------------------------------
template <class T>
__global__ void edges_cut (bool* mask,
                           const T* weights, T cut, int size){
//---------------------------------------------------------------------------------------------------------------------
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= size) return;

  mask[i] = (weights[i] > cut);
}

//---------------------------------------------------------------------------------------------------------------------
__global__ void first_vertex (bool* first_vertex,
                              const int* ivertex, const int* overtex, int size){
//---------------------------------------------------------------------------------------------------------------------
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= size) return;

  bool first_hit = true;
  int p = ivertex[i];

  for (int q=0; q<size; q++)
    if (p == overtex[q])
      first_hit = false;

  first_vertex[i] = first_hit;
}

//---------------------------------------------------------------------------------------------------------------------
__global__ void end_vertex (bool* first_vertex, bool* last_vertex,
                            const int* ivertex, const int* overtex, int size){
//---------------------------------------------------------------------------------------------------------------------
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= size) return;

  bool first_hit = true;
  int p = ivertex[i];
  for (int q=0; q<size; q++)
    if (p == overtex[q])
     first_hit = false; //last_vertex[q] = false;

  first_vertex[i] = first_hit;
}

//---------------------------------------------------------------------------------------------------------------------
__global__ void next_vertex (bool *next_vertex_mask,
                             const int* layer_overtex, int layer_size, const int* edges_ivertex, int edges_size){
//---------------------------------------------------------------------------------------------------------------------
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= edges_size) return;

  int ivertex = edges_ivertex[i];
  bool new_vertex = false;
  for (int q=0; q<layer_size; q++)
    if (layer_overtex[q] == ivertex)
      new_vertex = true;

  next_vertex_mask[i] = new_vertex;
}

//---------------------------------------------------------------------------------------------------------------------
__global__ void cuda_layers_duplicate_edges (bool *duplicates, bool *L1_mask, bool *L2_mask,
                                             int L1_nedges, const int *L1_source, const int* L1_target,
                                             int L2_nedges, const int *L2_source, const int *L2_target){
//---------------------------------------------------------------------------------------------------------------------
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= L1_nedges || j >= L2_nedges) return;

  if (L1_source[i] == L2_source[j]  &&  L1_target[i] == L2_target[j]) {
    L1_mask[i] = L2_mask[j] = false;
    *duplicates = true;
    printf ("DUPLICATE EDGE FOUND ON 2 DIFFERENT LAYERS \n");
  }
}

//---------------------------------------------------------------------------------------------------------------------
__global__ void tag_hits (bool *hits_in_tracks,
                          const int *source, const int* target, int nb_hits){
//---------------------------------------------------------------------------------------------------------------------
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nb_hits) return;

  hits_in_tracks[source[i]] = true;
  hits_in_tracks[target[i]] = true;
}

//---------------------------------------------------------------------------------------------------------------------
__global__ void tag_hits (bool *hits_in_tracks,
                          const int *source, const int* target, const int *tag, int nb_hits){
//---------------------------------------------------------------------------------------------------------------------
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nb_hits || !tag[i]) return;

  hits_in_tracks[source[i]] = true;
  hits_in_tracks[target[i]] = true;
}

//---------------------------------------------------------------------------------------------------------------------
__global__ void filter_last_edges (bool *mask,
                                   const int *positions, const bool *last_edges_mask, int nb_edges){
//---------------------------------------------------------------------------------------------------------------------
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nb_edges) return;

  mask[i] = mask[i] && last_edges_mask[positions[i]];
  //if (mask[i] && last_edges_mask[i]) printf ("i=%d last edge \n", i);
}

//---------------------------------------------------------------------------------------------------------------------
__global__ void filter_existing_hits (bool *mask,
                                      const int *hits, const bool *hits_in_tracks, int nb_edges, int shift){
//---------------------------------------------------------------------------------------------------------------------
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nb_edges) return;

  if (hits_in_tracks[hits[shift+i]]) mask[shift+i] = false;
  // if targets is from an edge of an existing track, mask[i] is already set to 0 (reverse mask)
}

//---------------------------------------------------------------------------------------------------------------------
__global__ void untag_last_edges (bool *last_edges_mask,
                                  const int *positions, const int *tags, int nb_edges){
//---------------------------------------------------------------------------------------------------------------------
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nb_edges) return;
  if (!tags[i]) return;

  last_edges_mask[positions[i]] = false;
}

//---------------------------------------------------------------------------------------------------------------------
__global__ void tag_tags (bool *mask,
                          const int *tag, int n){
//---------------------------------------------------------------------------------------------------------------------
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  mask[tag[i]] = true;
}

//---------------------------------------------------------------------------------------------------------------------
__global__ void indices2hitID (int *indices,
                               const int *hitID, int nb_vertices, int size){
//---------------------------------------------------------------------------------------------------------------------
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= size) return;

  int p = indices[i];
  if (p >= nb_vertices) return;
  indices[i] = hitID[p];
}


//=====CPU functions for edge layer connector========================================

//---------------------------------------------------------------------------------------------------------------------
template <class T>
CUDA_edge_layer_connector<T>::CUDA_edge_layer_connector (const boost::program_options::variables_map &po_vm){
//---------------------------------------------------------------------------------------------------------------------
  _nb_layers = 0;

  double total_time = 0;
  int nb_events = 0;
  _blocks = po_vm["gpu-nb-blocks"].as<int>();
  _events_dir = po_vm["input-dir"].as<std::string>() + "/";
  _event_filename_pattern = po_vm["input-filename-pattern"].as<std::string> ();
  _output_dir = po_vm["output-dir"].as<std::string>();
  _module_map_dir = po_vm["input-module-map"].template as<std::string>();
  _weights_cut = po_vm["edge-layer-connector-weights-cut"].as<T>();
  _min_hits = po_vm["edge-layer-connector-min-hits"].as<int>();
  _max_layers = po_vm["max-hits-per-track"].as<int>(); // default value: 30 layers of hits
  int device = po_vm["gpu-core"].as<int>();

  size_t f, t;
  cudaDeviceReset();
  cudaSetDevice(device);
  cudaMemGetInfo(&f, &t);
  std::cout << "CUDA infos: free memory " << f/pow(1024,3) << " Go / " << t/pow(1024,3) << "Go" << std::endl;

  // log all vertices events filenames
  std::string log_vertices_filenames = "events-vertices.txt";
  std::string cmd_vertices = "ls " + _events_dir + " |grep hit |grep "
                                   + _event_filename_pattern + " > " + log_vertices_filenames;
  std::cout << "cmd = " << cmd_vertices << std::endl;
  system(cmd_vertices.c_str());

  // log all edges events filenames
  std::string log_edges_filenames = "events-edges.txt";
  std::string cmd_edges = "ls " + _events_dir + " |grep edge |grep "
                                + _event_filename_pattern + " > " + log_edges_filenames;
  std::cout << "cmd = " << cmd_edges << std::endl;
  system(cmd_edges.c_str());

  // log all tracks events filenames
  std::string log_tracks_filenames = "events-tracks.txt";
  std::string cmd_tracks = "ls " + _events_dir + " |grep track |grep "
                                 + _event_filename_pattern + " > " + log_tracks_filenames;
  system(cmd_tracks.c_str());

  // open files containing events filenames
  std::ifstream v_file (log_vertices_filenames);
  if (v_file.fail())
    throw std::invalid_argument("Cannot open file " + log_vertices_filenames);

  std::ifstream e_file (log_edges_filenames);
  if (e_file.fail())
    throw std::invalid_argument("Cannot open file " + log_edges_filenames);

  std::ifstream t_file (log_tracks_filenames);
  if (t_file.fail())
    throw std::invalid_argument("Cannot open file " + log_tracks_filenames);

  std::string vertices_filename, edges_filename, tracks_filename;
  v_file >> vertices_filename;
  e_file >> edges_filename;
  t_file >> tracks_filename;

  // warm up GPU card
  cudaDeviceSynchronize();
  std::chrono::system_clock::time_point start_null = std::chrono::system_clock::now();
  null_kernel<<<1,1>>> ();
  cudaDeviceSynchronize();
  std::chrono::system_clock::time_point end_null = std::chrono::system_clock::now();
  std::cout << red << "init card cpu time : " << std::chrono::duration_cast<std::chrono::microseconds>(end_null - start_null).count() * 0.001 << reset;

  for (; !e_file.eof() && !v_file.eof() && !t_file.eof(); nb_events++) {
    std::cout << "vertices filename = " << vertices_filename << std::endl;
    std::cout << "edges filename = " << edges_filename << std::endl;
    std::cout << "tracks filename = " << tracks_filename << std::endl;
    _G = new CUDA_graph<T> (_events_dir, vertices_filename, edges_filename);

    std::cout << "start time" << std::endl;
    cudaDeviceSynchronize();
    std::chrono::system_clock::time_point start_time = std::chrono::system_clock::now();

    _edge_layers = new CUDA_edges<T> [_max_layers];
    build_tracks ();
    (*_tracks).filter (_min_hits, _blocks);
 // _edge_layers[0].layer_tag() starts at 1 (0->n = n elements) / tag = 0 not a track or deleted edge

    cudaDeviceSynchronize();
    std::chrono::system_clock::time_point end_time = std::chrono::system_clock::now();

    int nb_reconstructed_tracks = _edge_layers[0].layer_tag();
    std::cout << "nb tags = " << nb_reconstructed_tracks << reset;

    // get memory info
    size_t free_memory, total_memory;
    cudaMemGetInfo (&free_memory, &total_memory);

    total_time += std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() * 0.001;
    std::cout << red << "edge layer connector cpu time : " << std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() * 0.001 << " ms"<< reset;
    std::cout << "-----------------------------------------------------------------------------------" << std::endl;
//    CUDA_graph<T> G(_events_dir, vertices_filename, edges_filename);
//    incoming_vertex(G);
///    std::cout << "end step" << std::endl;
///    std::cout << "nb layers = " << nb_layers << std::endl;

    std::string filename = _output_dir + "/tracks-" + edges_filename; // pass as parameter when calling executable
    std::string filename_real = _output_dir + "/real-tracks-" + edges_filename; // pass as parameter when calling executable
    std::cout << "save tracks in " << filename << std::endl;
    save (filename);

    std::ofstream outfile;
    outfile.open("excel-edge-layer-connector-data.txt", std::ios_base::app);

    outfile << std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count()*0.001 << ","
            << (total_memory - free_memory)/pow(1024,3) << ","
            << (*_edges).nb_edges() << ","
            << _edge_layers[0].layer_tag() << ","
            << _nb_layers
            << std::endl;
    outfile.close();

/*
   // load real tracks
    tracks real_tracks;
    real_tracks.read_vertices_csv (_events_dir, vertices_filename);
    real_tracks.read_edges_csv (_events_dir, tracks_filename);
std::cout << "real edges event dir = " << _events_dir << std::endl;
std::cout << "real edges filename = " << vertices_filename << std::endl;
std::cout << "real tracks filename = " << tracks_filename << std::endl;

    // rebuild tracks
    real_tracks.build (_min_hits);

    // send tracks to GPU
    CUDA_tracks<T> cuda_real_tracks (real_tracks);
    real_tracks.save (_min_hits, filename_real);

    // convert indices to hitID
    int tracks_size = cuda_real_tracks.size();
    dim3 grid = (tracks_size + _blocks - 1) / _blocks;
    indices2hitID<<<grid,_blocks>>> (cuda_real_tracks.hits(), _G->hit_id(), _G->vertices(), tracks_size);

    int* nb_hits_sum;
    cudaMalloc (&nb_hits_sum, (real_tracks.nb_tracks()+1) * sizeof(int));
    cudaMemcpy (nb_hits_sum, real_tracks.hits_by_track_sum().data(), (real_tracks.nb_tracks()+1)*sizeof(int), cudaMemcpyHostToDevice);

    for (int i=1; i<=10; i++)
      std::cout << "last_tracks size = " << real_tracks.nb_hits_by_track()[real_tracks.nb_tracks()-i] << std::endl;

    CUDA_mask tracks_match ((*_tracks).size());
    int *tracks_positions;
    std::cout << "nb reconst. tracks = " << nb_reconstructed_tracks << " vs " << (*_tracks).nb_tracks() << std::endl;
    std::cout << "nb real tracks = " << real_tracks.nb_tracks() << std::endl;
    cudaMalloc (&tracks_positions, nb_reconstructed_tracks * sizeof(int));
//    dim3 grid = (nb_reconstructed_tracks + _blocks - 1) / _blocks;
    grid = (nb_reconstructed_tracks + _blocks - 1) / _blocks;
    init_vector<<<grid,_blocks>>> (tracks_positions, -1, nb_reconstructed_tracks);

    // compare build tracks with real tracks
    int max_hits_per_tracks = _nb_layers + 1; // n layers = n edges = n+1 hits
    CUDA_mask real_tracks_match (cuda_real_tracks.size());

    int blocks_2d = sqrt (_blocks);
    dim3 blocks_dim_2d (blocks_2d,blocks_2d);
    dim3 grid_dim_2d ((nb_reconstructed_tracks + blocks_2d - 1) / blocks_2d, (real_tracks.nb_tracks() + blocks_2d - 1) / blocks_2d);

    print_track<<<1,1>>> ((*_tracks).nb_hits(), (*_tracks).hits(), 2, (*_tracks).max_hits_per_track ());
    print_track<<<1,1>>> ((*_tracks).nb_hits(), (*_tracks).hits(), 7, (*_tracks).max_hits_per_track ());

    associate_reconstructed_real_tracks<<<grid_dim_2d,blocks_dim_2d>>> (tracks_positions, (*_tracks).hits(), (*_tracks).nb_hits(), cuda_real_tracks.hits(), nb_hits_sum, max_hits_per_tracks, _min_hits, nb_reconstructed_tracks, real_tracks.nb_tracks());

    dim3 grid_dim = (nb_reconstructed_tracks + _blocks - 1) / _blocks;
    compare_tracks <<<grid_dim,_blocks>>> (tracks_match.cuda_mask(), real_tracks_match.cuda_mask(), tracks_positions,
                                          (*_tracks).hits(), (*_tracks).nb_hits(), cuda_real_tracks.hits(), nb_hits_sum, max_hits_per_tracks, _min_hits, nb_reconstructed_tracks, real_tracks.nb_tracks());

    CUDA_mask perfect (nb_reconstructed_tracks);
    CUDA_mask partial (nb_reconstructed_tracks);
    int *true_positive, *false_positive, *false_negative;
    cudaMalloc (&true_positive, nb_reconstructed_tracks * sizeof(int));
    cudaMalloc (&false_positive, nb_reconstructed_tracks * sizeof(int));
    cudaMalloc (&false_negative, nb_reconstructed_tracks * sizeof(int));
    evaluate_track_reconstruction<<<grid_dim,_blocks>>> (true_positive, false_positive, false_negative, perfect.cuda_mask(), partial.cuda_mask(),
                                                         tracks_match.cuda_mask(), real_tracks_match.cuda_mask(), tracks_positions,
                                                        (*_tracks).hits(), (*_tracks).nb_hits(), cuda_real_tracks.hits(), nb_hits_sum,
                                                         max_hits_per_tracks, nb_reconstructed_tracks);

    filename_real = _output_dir + "/tracks-perfects.csv";
    std::cout << "saving perfects to " << filename_real << std::endl;
    int nb_perfects = perfect.scan (_blocks);
    (*_tracks).save (_min_hits, perfect, filename_real);

    for (int i=0; i<=nb_reconstructed_tracks; i++) {
      print_track_reconstruction <<<1,1>>> (i, tracks_match.cuda_mask(), real_tracks_match.cuda_mask(), tracks_positions, perfect.cuda_mask(),
                                           (*_tracks).hits(), (*_tracks).nb_hits(), cuda_real_tracks.hits(), nb_hits_sum, max_hits_per_tracks, _min_hits, nb_reconstructed_tracks);
//      print_statistics<<<1,1>>> (i, true_positive, false_positive, false_negative, tracks_positions, (*_tracks).nb_hits(), cuda_real_tracks.nb_hits(), nb_reconstructed_tracks);
    }

    int nb_partials = partial.scan (_blocks);
    std::cout << "nb first edges = " << _edge_layers[0].nb_edges() << std::endl;
    std::cout << "nb reconstructed tracks = " << nb_reconstructed_tracks << std::endl;
    std::cout << green << "nb perfects = " << nb_perfects << reset;
    std::cout << blue << "% of perfects in reconstructed = " << (float) nb_perfects / (float) nb_reconstructed_tracks * 100  << reset;
    std::cout << blue << "% of perfects in true tracks = " << (float) nb_perfects / (float) real_tracks.nb_tracks() * 100  << reset;
    std::cout << magenta << "% of partials in reconstructed = " << (float) nb_partials / (float) nb_reconstructed_tracks * 100  << reset;
    std::cout << magenta << "% of partials in true tracks = " << (float) nb_partials / (float) real_tracks.nb_tracks() * 100  << reset;
cudaDeviceSynchronize();
    cudaFree (nb_hits_sum);
    cudaFree (false_negative);
    cudaFree (false_positive);
    cudaFree (true_positive);
    cudaFree (tracks_positions);
//    cudaFree (_edges_position);
*/

    e_file >> edges_filename;
    v_file >> vertices_filename;
    t_file >> tracks_filename;
    delete _G;
    delete _edges;
    delete [] _edge_layers;
    delete _tracks;
    _G = nullptr;
    _edges = nullptr;
    _edge_layers = nullptr;
    _tracks = nullptr;
    _nb_layers = 0;
  }

  v_file.close();
  e_file.close();
  t_file.close();

  std::cout << blue << "total time for " << nb_events << " events = " << total_time << reset;
  std::cout << blue << "average time per event = " << total_time / nb_events << reset;
  // get memory info
  size_t free_memory, total_memory;
  cudaMemGetInfo (&free_memory, &total_memory);
  std::cout << blue << "CUDA infos: used memory " << (total_memory - free_memory)/pow(1024,3) << " Go / " << total_memory/pow(1024,3) << "Go" << reset;
  cudaDeviceSynchronize();
  GPU_ERROR_CHECK (cudaGetLastError());
//  module_map_triplet<T> module_map_triplet;
//  module_map_triplet.read_TTree(_module_map_dir.c_str());
//  if (!module_map_triplet)
//    throw std::runtime_error("Cannot retrieve ModuleMap from " + _module_map_dir);

//  _cuda_module_map_doublet = CUDA_module_map_doublet<T> (module_map_triplet);
//  _cuda_module_map_doublet.HostToDevice();
//  _cuda_module_map_triplet = CUDA_module_map_triplet<T> (module_map_triplet);
//  _cuda_module_map_triplet.HostToDevice();

//  std::cout << "# of modules = " << module_map_triplet.module_map().size() << std::endl;
}

//---------------------------------------------------------------------------------------------------------------------
template <class T>
CUDA_edge_layer_connector<T>::CUDA_edge_layer_connector (CUDA_graph<T>* G, const T& weights_cut, const int& min_hits,
                                                         int blocks, int max_hits_per_track){
//---------------------------------------------------------------------------------------------------------------------
  _blocks = blocks;
  _G = G;
  _weights_cut = weights_cut;
  _min_hits = min_hits;
  _max_layers = max_hits_per_track;
  _edge_layers = new CUDA_edges<T> [_max_layers]; // nb of hits layers
  _nb_layers = 0;
}

//---------------------------------------------------------------------------------------------------------------------
template <class T>
CUDA_edge_layer_connector<T>::~CUDA_edge_layer_connector (){
//---------------------------------------------------------------------------------------------------------------------
  if (_nb_layers) {
    delete _edges;
    delete [] _edge_layers;
    delete _tracks;
    _edges = nullptr;
    _edge_layers = nullptr;
    _tracks = nullptr;
    _G = nullptr;
    _nb_layers = 0;
  }
}

//---------------------------------------------------------------------------------------------------------------------
template <class T>
void CUDA_edge_layer_connector<T>::free (){
//---------------------------------------------------------------------------------------------------------------------
  if (_nb_layers) {
    delete _edges;
    delete [] _edge_layers;
    delete _tracks;
    _edges = nullptr;
    _edge_layers = nullptr;
    _tracks = nullptr;
    _G = nullptr;
    _nb_layers=0;
  }
}

//---------------------------------------------------------------------------------------------------------------------
template <class T>
void CUDA_edge_layer_connector<T>::build_tracks (){
//---------------------------------------------------------------------------------------------------------------------

  //---------------------
  // cut low edge weights
  //---------------------

  weight_cuts ();
  if (!_nb_edges) return;

  // define hit indices as not included in a layer
  int nb_hits = (*_G).vertices();

  // ---------------------------------
  // get 1st and last layer of edges
  // (from the center of the detector)
  // ---------------------------------

  first_edges ();
  if (!_edge_layers[0].nb_edges()) return;

//  CUDA_edges<T> cuda_last_edges;
//  last_edges(cuda_last_edges);
//  std::cout << "nb targets = " << cuda_last_edges.nb_edges() << " / "  << _nb_edges << reset;

  //------------------
  // build edge layers (output last edge vertex -> input next edge)
  //------------------

  _nb_layers = layers ();
  if (!_nb_layers) return;

  //-----------------------------------------
  // tag first layer edges from 0 to nb edges (close to the outer part of the detector)
  //-----------------------------------------

//  CUDA_mask mask (_edge_layers[_nb_layers-1].nb_edges());
//  _edge_layers[_nb_layers-1].filter_last_edges (mask, cuda_last_edges, _blocks);

//  print_vector<<<1,1>>> (cuda_last_edges.target(), cuda_last_edges.nb_edges());
//  print_edges<<<1,1>>> (cuda_last_edges.source(), cuda_last_edges.target(), cuda_last_edges.weight(), cuda_last_edges.tags(), xxx, cuda_last_edges.nb_edges());
  _edge_layers[_nb_layers-1].tag_edges (_blocks);

  //--------------------------
  // for each layer sort edges (as edge of an existing track vs new edges) and tag them
  //--------------------------

  for (int i=_nb_layers-2; i>=0; i--)
    //sort_v1 (_edge_layers[i], _edge_layers[i+1], i);
    //sort_v2 (_edge_layers[i], _edge_layers[i+1], i);
    //sort_v4 (_edge_layers[i], _edge_layers[i+1], i);
    sort_v5 (_edge_layers[i],
             _edge_layers[i+1], i);
//exit(0);

  //---------------------------------------------------
  // mark the tags which reach the first layer as valid (others are not valid => kill outliers)
  //---------------------------------------------------

//  int max_tags = _edge_layers[0].layer_tag();
//  CUDA_mask tag_mask (max_tags+1);
//  dim3 grid = (max_tags + _blocks - 1) / _blocks;
//  tag_tags<<<grid,_blocks>>> (tag_mask.cuda_mask(),
//                              _edge_layers[0].tags(), max_tags);

  //----------------------
  // tracks reconstruction
  //----------------------

  int nb_reconstructed_tracks = _edge_layers[0].layer_tag();
//  int nb_reconstructed_tracks = tag_mask.scan(_blocks);
  _tracks = new CUDA_tracks<T> (nb_reconstructed_tracks, _nb_layers);

  // copy layers to tracks (columns with same tag -> track as a line)

  for (int i=0; i<_nb_layers; i++) {
    _edge_layers[i].untag_duplicate_targets (_blocks);
    (*_tracks).add (_edge_layers[i], _blocks);
  }

  //-------------------------
  // convert indices to hitID
  //-------------------------

  int tracks_size = _tracks->size();
  dim3 grid = (tracks_size + _blocks - 1) / _blocks;
  indices2hitID<<<grid,_blocks>>> (_tracks->hits(), _G->hit_id(), _G->vertices(), tracks_size);

  //--------------------------
  // for each layer sort edges (as edge of an existing track vs new edges) and tag them
  //--------------------------
//  for (int i=nb_layers-2; i>=0; i--) {
//    if (i+1 < _min_hits) _edge_layers[i].cut_new_edges (_edge_layers[i+1], _blocks);
//    sort (_edge_layers[i], _edge_layers[i+1], last_edges_mask, i);
//  }

//  // ----------------------
//  // get 1st layer of edges
//  // ----------------------
//  first_edges (_edge_layers[0]); // (from the center of the detector)

/*
  // ---------------------------------
  // get 1st and last layer of edges
  // (from the center of the detector)
  // ---------------------------------

  int nb_edges = (*_edges).nb_edges();
  CUDA_edges<T> last_edges;
  end_edges (_edge_layers[0], last_edges);

  //------------------
  // build edge layers (output last edge vertex -> input next edge)
  //------------------
  int nb_layers = layers ();
//  std::cout << "nb layers = " << nb_layers << std::endl;
//  std::cout << "nb filtered edges = " << (*_edges).nb_edges() << std::endl;

  //-----------------------------------------
  // tag first layer edges from 0 to nb edges (close to the outer part of the detector)
  //-----------------------------------------

  _edge_layers[nb_layers-1].tag_edges (_blocks);

  _factor = 1;
//  print_colored_vector<<<1,1>>> (_edge_layers[nb_layers-1].target(), _edge_layers[nb_layers-1].nb_edges(), _edge_layers[nb_layers-1].nb_edges());
//  print_colored_vector<<<1,1>>> (_edge_layers[nb_layers-1].source(), _edge_layers[nb_layers-1].nb_edges(), _edge_layers[nb_layers-1].nb_edges());
//  print_colored_double_vector<<<1,1>>> (_edge_layers[nb_layers-1].weight(), _edge_layers[nb_layers-1].nb_edges(), _edge_layers[nb_layers-1].nb_edges());
//  print_colored_vector<<<1,1>>> (_edge_layers[nb_layers-1].tags(), _edge_layers[nb_layers-1].nb_edges(), _edge_layers[nb_layers-1].nb_edges());

  //--------------------------
  // for each layer sort edges (as edge of an existing track vs new edges) and tag them
  //--------------------------

  for (int i=nb_layers-2; i>=0; i--) {
    if (i+1 < _min_hits) _edge_layers[i].cut_new_edges (_edge_layers[i+1], _blocks);
    sort (_edge_layers[i], _edge_layers[i+1], last_edges);
  }

    int nb_reconstructed_tracks = _edge_layers[0].layer_tag(); // _edge_layers[0].layer_tag() starts at 1 (0->n = n elements) / tag = 0 not a track or deleted edge
    _tracks = new CUDA_tracks (nb_reconstructed_tracks, nb_layers);

    // copy layers to tracks (columns with same tag -> track as a line)
    for (int i=0; i<nb_layers; i++) {
      _edge_layers[i].untag_duplicate_targets (_factor, _blocks);
      (*_tracks).add (_edge_layers[i], _blocks);
    }
*/
  cudaDeviceSynchronize();
  GPU_ERROR_CHECK (cudaGetLastError());
}

//---------------------------------------------------------------------------------------------------------------------
// cuts on edge weight (weights are issued by the GNN algorithm)
template <class T>
void CUDA_edge_layer_connector<T>::weight_cuts (){
//---------------------------------------------------------------------------------------------------------------------
  int nb_edges = (*_G).size();
  dim3 grid_dim = (nb_edges + _blocks - 1) / _blocks;

  CUDA_mask mask (nb_edges);
  edges_cut<<<grid_dim,_blocks>>> (mask.cuda_mask(), (*_G).weights(), _weights_cut, nb_edges);
  _edges = new CUDA_edges<T> ((*_G).edges(), mask, _blocks);
  _nb_edges = (*_edges).nb_edges();
}

//---------------------------------------------------------------------------------------------------------------------
// extract first vertices
template <class T>
void CUDA_edge_layer_connector<T>::first_edges (){
//---------------------------------------------------------------------------------------------------------------------
  CUDA_mask cuda_first_vertex_mask (_nb_edges);
  int blocks = 64;

  dim3 grid = (_nb_edges + blocks - 1) / blocks;
  first_vertex<<<grid,blocks>>> (cuda_first_vertex_mask.cuda_mask(),
                                (*_edges).source(), (*_edges).target(), _nb_edges);

  // build 1st edges
  _edge_layers[0].build (_edges, cuda_first_vertex_mask, _blocks);
}

//---------------------------------------------------------------------------------------------------------------------
// extract last vertices
template <class T>
void CUDA_edge_layer_connector<T>::last_edges (CUDA_edges<T>& last_edges){
//---------------------------------------------------------------------------------------------------------------------
  CUDA_mask cuda_last_edges_mask (_nb_edges);
  int blocks = 64;

  dim3 grid = (_nb_edges + blocks - 1) / blocks;
  first_vertex<<<grid,blocks>>> (cuda_last_edges_mask.cuda_mask(),
                                (*_edges).target(), (*_edges).source(), _nb_edges);

  last_edges.build (_edges, cuda_last_edges_mask, _blocks);

/*
  int nb_edges = (*_edges).nb_edges();
  dim3 grid_dim = (nb_edges + _blocks - 1) / _blocks;
  CUDA_mask cuda_last_vertex_mask;
  cuda_last_vertex_mask.init (nb_edges, true, _blocks);
  std::cout << "nb total edges before selection = " << nb_edges << std::endl;

  int blocks_2d = sqrt (2 * _blocks);
  int size_2d = (nb_edges + blocks_2d - 1) / blocks_2d;
  dim3 block_dim_2d (blocks_2d, blocks_2d);
  dim3 grid_dim_2d (size_2d, size_2d);
  first_vertex<<<grid_dim_2d,block_dim_2d>>> (cuda_last_vertex_mask.cuda_mask(),
                                             (*_edges).target(), (*_edges).source(), nb_edges);

  //-------------
  // reduce edges
  //-------------

  CUDA_mask cuda_other_vertex_mask (nb_edges);
  reverse_mask<<<grid_dim,_blocks>>> (cuda_other_vertex_mask.cuda_mask(),
                                      cuda_last_vertex_mask.cuda_mask(), nb_edges);
  CUDA_edges<T> cuda_other_vertex (_edges, cuda_other_vertex_mask, _blocks);
///  CUDA_edges<T> cuda_other_vertex;
///  cuda_other_vertex.reverse_build (_edges, cuda_other_vertex_mask, _blocks);

  int nb_other_vertices = cuda_other_vertex.nb_edges();

  //------------------
  // build first edges
  //------------------

  cuda_first_edges.build (_edges, cuda_last_vertex_mask, _blocks);

  std::cout << blue << "nb 1st vertices = " << cuda_first_edges.nb_edges() << reset;
  std::cout << "# other vertices = " << nb_other_vertices << std::endl;

  (*_edges).reduce (cuda_other_vertex_mask, _blocks);
//  (*_edges).reverse_reduce (cuda_last_vertex_mask, _blocks);

  std::cout << "nb edges = " << (*_edges).nb_edges() << std::endl;
  std::cout << "end 1st edges" << std::endl;
*/
}

//---------------------------------------------------------------------------------------------------------------------
// extract end vertices (first and last)
template <class T>
void CUDA_edge_layer_connector<T>::end_edges (CUDA_mask& cuda_last_edges_mask){
//---------------------------------------------------------------------------------------------------------------------
//  dim3 grid_dim = (_nb_edges + _blocks - 1) / _blocks;
  CUDA_mask edge_masks (_nb_edges);
  int blocks = 64;
  cuda_last_edges_mask.init (_nb_edges, true, _blocks);

  dim3 grid = (_nb_edges + blocks - 1) / blocks;
  end_vertex<<<grid,blocks>>> (edge_masks.cuda_mask(), cuda_last_edges_mask.cuda_mask(),
                              (*_edges).source(), (*_edges).target(), _nb_edges);

  // build 1st edges
  _edge_layers[0].build (_edges, edge_masks, _blocks);
}

//---------------------------------------------------------------------------------------------------------------------
// extract next edge
template <class T>
void CUDA_edge_layer_connector<T>::next_edges (CUDA_mask& cuda_next_vertex_mask, int step){
//---------------------------------------------------------------------------------------------------------------------
  // find next edges in _edges
  int blocks = 32;
  dim3 grid = (_nb_edges + blocks - 1) / blocks;
  next_vertex<<<grid,blocks>>> (cuda_next_vertex_mask.cuda_mask(),
                                _edge_layers[step].target(), _edge_layers[step].nb_edges(), (*_edges).source(), _nb_edges);

  // build next edges
  _edge_layers[step+1].build (_edges, cuda_next_vertex_mask, _blocks); // build the new layer of edges with a link to the global edges position
}

/*
//----------------------------------------------------------------------------------------------------------------
// extract previous edge
template <class T>
void CUDA_edge_layer_connector<T>::previous_edges (CUDA_edges<T>& cuda_next_edges, CUDA_edges<T>& cuda_prev_edges)
//----------------------------------------------------------------------------------------------------------------
{
  int blocks_2d = sqrt (2 * _blocks);
  dim3 block_dim_2d (blocks_2d, blocks_2d);
  int nb_vertices = (*_edges).nb_edges();
  int nb_prev_vertices = cuda_prev_edges.nb_edges();
  int size_vtxo = (nb_vertices + blocks_2d - 1) / blocks_2d;
  int size_vtx1 = (nb_prev_vertices + blocks_2d - 1) / blocks_2d;
  dim3 grid_dim_2d_vtx2 (size_vtx1, size_vtxo);
  CUDA_mask cuda_next_vertex_mask (nb_vertices);
  next_vertex<<<grid_dim_2d_vtx2,block_dim_2d>>> (cuda_next_vertex_mask.cuda_mask(), cuda_prev_edges.source(), (*_edges).target(), nb_prev_vertices, nb_vertices);

  cuda_next_edges.build (_edges, cuda_next_vertex_mask, _blocks);
}
*/

//---------------------------------------------------------------------------------------------------------------------
// build layers of edges based on layer edge target -> edge source of next layer
template <class T>
int CUDA_edge_layer_connector<T>::layers (){
//---------------------------------------------------------------------------------------------------------------------
  int step = 0;
  CUDA_mask cuda_next_vertex_mask (_nb_edges);

  for (; step < _max_layers-1 && _edge_layers[step].nb_edges(); step++){
    next_edges (cuda_next_vertex_mask, step);
  }

  if (step == _max_layers - 1){
    std::cout << magenta << "WARNING: max hits per track reached \nnb of edges in the last layer = " << _edge_layers[step].nb_edges() << reset;
    if (_edge_layers[step].nb_edges() == 1 && _edge_layers[step-1].nb_edges() == 1)
    std::cout << red << "possible loop detected in data" << reset;
  }

  return step;
}

//---------------------------------------------------------------------------------------------------------------------
// extract end vertices (first and last)
template <class T>
void CUDA_edge_layer_connector<T>::layers_duplicate_edges (CUDA_edges<T> *edge_layers, int nb_layers){
//---------------------------------------------------------------------------------------------------------------------
  int blocks_2d = sqrt (2 * _blocks);
  dim3 block_dim_2d (blocks_2d, blocks_2d);
  bool *duplicates;
  cudaMalloc (&duplicates, sizeof(bool));

  for (int i=1; i<nb_layers-1; i++) {  // first layer cannot contain other layer edges (they would be in the first layer - layer with no antecedent)
    int L1_nb_edges = edge_layers[i].nb_edges();
    int size1 = (L1_nb_edges + blocks_2d - 1) / blocks_2d;
    CUDA_mask L1_mask (L1_nb_edges);
    dim3 grid_dim = ((L1_nb_edges + _blocks - 1) / _blocks);
    init_vector<<<grid_dim,_blocks>>> (L1_mask.cuda_mask(), false, L1_nb_edges);

    for (int j=i+1; j<nb_layers; j++) {
      int L2_nb_edges = edge_layers[j].nb_edges();
      int size2 = (L2_nb_edges + blocks_2d - 1) / blocks_2d;
      CUDA_mask L2_mask (L2_nb_edges);
      grid_dim = ((L2_nb_edges + _blocks - 1) / _blocks);
      init_vector<<<grid_dim,_blocks>>> (L2_mask.cuda_mask(), true, L2_nb_edges);

      dim3 grid_dim_2d (size1, size2);
      cuda_layers_duplicate_edges<<<grid_dim_2d,_blocks>>> (duplicates, L1_mask.cuda_mask(), L2_mask.cuda_mask(),
                                                            L1_nb_edges, edge_layers[i].source(), edge_layers[i].target(),
                                                            L2_nb_edges, edge_layers[j].source(), edge_layers[j].target());
    }
  }

  cudaFree (duplicates);
}

//-------------------------------------------------------------------------------------------------
template <class T>
void CUDA_edge_layer_connector<T>::sort_v2 (CUDA_edges<T>& layer,
                                            CUDA_edges<T>& next_layer, int layer_nb){
//-------------------------------------------------------------------------------------------------
  int layer_nb_edges = layer.nb_edges();
  dim3 grid_dim ((layer_nb_edges + _blocks - 1) / _blocks);
  CUDA_mask edges_mask (layer_nb_edges);

  //--------------------------------------------------------------------
  // sort in layer the edges connected to next_layer edges and new edges
  //--------------------------------------------------------------------

  // create edges mask with 0 if layer.target is not in next_layer.source (new egde) and 1 if layer.target is in next_layer.source (existing edge)
  layer.connect_layer_edges (edges_mask, next_layer, _blocks);

  // untag duplicate sources
  CUDA_mask duplicate_sources (layer_nb_edges);
  layer.find_duplicate_sources (duplicate_sources, edges_mask, _blocks);

  // modify filter to avoid duplicate edges
  mask_filter<<<grid_dim,_blocks>>> (edges_mask.cuda_mask(),
                                     duplicate_sources.cuda_mask(), layer_nb_edges);

  // create layer_connected_edges from layer: separate layer edges (connected to next layer edges) and new edges
  CUDA_edges<T> layer_connected_edges (&layer, edges_mask, _blocks); // layer_connected_edges <-> (layer target edges == next layer source edges)

  // get edges layer tag from next_layer edges tag
  layer_connected_edges.copy_tags (next_layer, _blocks);

  CUDA_edges<T> new_edges;

  if (layer_nb+2 >= _min_hits){ // layer 0 = 2 hits, layer 1 = 3 hits, layer 2 = 4 hits, layer 3 = 5 hits ...

    // kill edges with duplicate sources (based on edges weigths - except if edge is linked to an edge of the next layer)
    mask_reverse_filter<<<grid_dim,_blocks>>> (edges_mask.cuda_mask(), duplicate_sources.cuda_mask(), layer_nb_edges);

//    CUDA_mask duplicate_sources (layer_nb_edges);
//    layer.untag_duplicate_sources (duplicate_sources, edges_mask, _blocks);

    // kill new edges when they have a duplicate source with an existing track

    // get new edges on layer
    new_edges.reverse_build (&layer, edges_mask, _blocks);

///    new_edges.reverse_build (&layer, duplicate_sources, _blocks);
//    new_edges.tag_edges (_blocks, layer_connected_edges.layer_tag());

//    CUDA_mask duplicates_mask (new_edges.nb_edges());
//    new_edges.reverse_reduce (duplicates_mask, _blocks);
    new_edges.tag_edges (_blocks, layer_connected_edges.layer_tag());
  }

  // sort edges by tag (=> edges are ordered by track)
  int *sorted_vector_positions;
  cudaMalloc (&sorted_vector_positions, layer_connected_edges.nb_edges() * sizeof(int));
  thrust_sort (sorted_vector_positions, layer_connected_edges.tags(), layer_connected_edges.nb_edges(), _blocks);

  layer.merge_edges (layer_connected_edges, new_edges, sorted_vector_positions, _blocks); // pb with sorted_vector size ?
//  layer.untag_duplicate_sources (_blocks);
  cudaFree (sorted_vector_positions);
}

//-------------------------------------------------------------------------------------------------
template <class T>
void CUDA_edge_layer_connector<T>::sort_v1 (CUDA_edges<T>& layer,
                                            CUDA_edges<T>& next_layer, int layer_nb)
//-------------------------------------------------------------------------------------------------
{
  int layer_nb_edges = layer.nb_edges();
  dim3 grid_dim ((layer_nb_edges + _blocks - 1) / _blocks);
  CUDA_mask edges_mask (layer_nb_edges);

  //--------------------------------------------------------------------
  // sort in layer the edges connected to next_layer edges and new edges
  //--------------------------------------------------------------------

  // create edges mask with 0 if layer.target is not in next_layer.source (new egde) and 1 if layer.target is in next_layer.source (existing edge)
  layer.connect_layer_edges (edges_mask, next_layer, _blocks);

  // create layer_connected_edges from layer: separate layer edges (connected to next layer edges) and new edges
  CUDA_edges<T> layer_connected_edges (&layer, edges_mask, _blocks); // layer_connected_edges <-> (layer target edges == next layer source edges)

  // get edges layer tag from next_layer edges tag
  layer_connected_edges.copy_tags (next_layer, _blocks); // MODIFY this part: _factor is not used and edge tag is aleatory (and so not deterministic)

  // sort edges by tag (=> edges are ordered by track)
  int *sorted_vector_positions;
  cudaMalloc (&sorted_vector_positions, layer_connected_edges.nb_edges() * sizeof(int));
  thrust_sort (sorted_vector_positions, layer_connected_edges.tags(), layer_connected_edges.nb_edges(), _blocks);

  CUDA_edges<T> new_edges;

  if (layer_nb+2 >= _min_hits){ // layer 0 = 2 hits, layer 1 = 3 hits, layer 2 = 4 hits, layer 3 = 5 hits ...
    // get new edges on layer
    new_edges.reverse_build (&layer, edges_mask, _blocks);
    new_edges.tag_edges (_blocks, layer_connected_edges.layer_tag());
  }

  layer.merge_edges (layer_connected_edges, new_edges, sorted_vector_positions, _blocks); // pb with sorted_vector size ?
  layer.untag_duplicate_sources (_blocks);

  cudaFree (sorted_vector_positions);
/*
if (layer_nb > 17){
  print_vector<<<1,1>>> (layer.target(), layer.nb_edges());
  print_vector<<<1,1>>> (layer.source(), layer.nb_edges());
  print_vector<<<1,1>>> (layer.tags(), layer.nb_edges());
}
*/
//print_edge<<<1,1>>> (layer.source(), layer.target(), layer.weight(), layer.tags(), 44820, layer.nb_edges());
//print_edge<<<1,1>>> (layer.source(), layer.target(), layer.weight(), layer.tags(), 118239, layer.nb_edges());
print_edge_tag<<<1,1>>> (layer.source(), layer.target(), layer.weight(), layer.tags(), 68, layer.nb_edges());
print_edge_tag<<<1,1>>> (layer.source(), layer.target(), layer.weight(), layer.tags(), 118, layer.nb_edges());
}

//-------------------------------------------------------------------------------------------------
template <class T>
void CUDA_edge_layer_connector<T>::sort_v3 (CUDA_edges<T>& layer,
                                            CUDA_edges<T>& next_layer, int layer_nb){
//-------------------------------------------------------------------------------------------------
  int layer_nb_edges = layer.nb_edges();
  dim3 grid_dim ((layer_nb_edges + _blocks - 1) / _blocks);
  CUDA_mask edges_mask (layer_nb_edges);

  //--------------------------------------------------------------------
  // sort in layer the edges connected to next_layer edges and new edges
  //--------------------------------------------------------------------

  // create edges mask with 0 if layer.target is not in next_layer.source (new egde) and 1 if layer.target is in next_layer.source (existing edge)
  layer.connect_layer_edges (edges_mask, next_layer, _blocks);

  // untag duplicate sources
  CUDA_mask duplicate_sources (layer_nb_edges);
  layer.find_duplicate_sources (duplicate_sources, edges_mask, _blocks);

  // modify filter to avoid duplicate edges
//  mask_filter<<<grid_dim,_blocks>>> (edges_mask.cuda_mask(),
//                                     duplicate_sources.cuda_mask(), layer_nb_edges);

  // create layer_connected_edges from layer: separate layer edges (connected to next layer edges) and new edges
  CUDA_edges<T> layer_connected_edges (&layer, edges_mask, _blocks); // layer_connected_edges <-> (layer target edges == next layer source edges)

  // get edges layer tag from next_layer edges tag
  layer_connected_edges.copy_tags (next_layer, _blocks);

  CUDA_edges<T> new_edges;

  if (layer_nb+2 >= _min_hits){ // layer 0 = 2 hits, layer 1 = 3 hits, layer 2 = 4 hits, layer 3 = 5 hits ...

    edges_mask.reverse (_blocks);

    // kill edges with duplicate sources (based on edges weigths - except if edge is linked to an edge of the next layer)
    mask_filter<<<grid_dim,_blocks>>> (edges_mask.cuda_mask(),
                                       duplicate_sources.cuda_mask(), layer_nb_edges);

//    CUDA_mask duplicate_sources (layer_nb_edges);
//    layer.untag_duplicate_sources (duplicate_sources, edges_mask, _blocks);

    // kill new edges when they have a duplicate source with an existing track

    // get new edges on layer
    new_edges.build (&layer, edges_mask, _blocks);

///    new_edges.reverse_build (&layer, duplicate_sources, _blocks);
    new_edges.tag_edges (_blocks, layer_connected_edges.layer_tag());

//    CUDA_mask duplicates_mask (new_edges.nb_edges());
//    new_edges.reverse_reduce (duplicates_mask, _blocks);
///    new_edges.tag_edges (_blocks, next_layer_edges.layer_tag());
  }

  // sort edges by tag (=> edges are ordered by track)
  int *sorted_vector_positions;
  cudaMalloc (&sorted_vector_positions, layer_connected_edges.nb_edges() * sizeof(int));
  thrust_sort (sorted_vector_positions, layer_connected_edges.tags(), layer_connected_edges.nb_edges(), _blocks);

  layer.merge_edges (layer_connected_edges, new_edges, sorted_vector_positions, _blocks); // pb with sorted_vector size ?
  layer.untag_duplicate_sources (_blocks);

  cudaFree (sorted_vector_positions);

/*
if (layer_nb > 17){
  print_vector<<<1,1>>> (layer.target(), layer.nb_edges());
  print_vector<<<1,1>>> (layer.source(), layer.nb_edges());
  print_vector<<<1,1>>> (layer.tags(), layer.nb_edges());
}
*/
}

//v4 = v2 @20251202
//-------------------------------------------------------------------------------------------------
template <class T>
void CUDA_edge_layer_connector<T>::sort_v4 (CUDA_edges<T>& layer,
                                            CUDA_edges<T>& next_layer, int layer_nb){
//-------------------------------------------------------------------------------------------------
  int layer_nb_edges = layer.nb_edges();
  dim3 grid_dim ((layer_nb_edges + _blocks - 1) / _blocks);
  CUDA_mask edges_mask (layer_nb_edges);
//if (!layer_nb) print_edge<<<1,1>>> (next_layer.source(), next_layer.target(), next_layer.weight(), 127254, next_layer.nb_edges());
//if (!layer_nb) print_edge<<<1,1>>> (next_layer.source(), next_layer.target(), next_layer.weight(), 127275, next_layer.nb_edges());
//if (!layer_nb) print_edge<<<1,1>>> (layer.source(), layer.target(), layer.weight(), 120222, layer.nb_edges());

  //--------------------------------------------------------------------
  // sort in layer the edges connected to next_layer edges and new edges
  //--------------------------------------------------------------------

  // create edges mask with 0 if layer.target is not in next_layer.source (new egde) and 1 if layer.target is in next_layer.source (existing edge)
  layer.connect_layer_edges (edges_mask, next_layer, _blocks);

  // untag duplicate sources
  CUDA_mask duplicate_sources (layer_nb_edges);
  layer.find_duplicate_sources (duplicate_sources, edges_mask, _blocks);

  // modify filter to avoid duplicate edges
//  mask_filter<<<grid_dim,_blocks>>> (edges_mask.cuda_mask(),
//                                     duplicate_sources.cuda_mask(), layer_nb_edges);

  // create layer_connected_edges from layer: separate layer edges (connected to next layer edges) and new edges
  CUDA_edges<T> layer_connected_edges (&layer, edges_mask, _blocks); // layer_connected_edges <-> (layer target edges == next layer source edges)

  // get edges layer tag from next_layer edges tag
  layer_connected_edges.copy_tags (next_layer, _blocks);

  // filter on cummulated weights existing edges
  layer_connected_edges.untag_duplicate_sources (_blocks);

  CUDA_edges<T> new_edges;

  if (layer_nb+2 >= _min_hits){ // layer 0 = 2 hits, layer 1 = 3 hits, layer 2 = 4 hits, layer 3 = 5 hits ...

    // kill new edges with duplicate sources (based on edges weigths - except if edge is linked to an edge of the next layer)
    mask_reverse_filter<<<grid_dim,_blocks>>> (edges_mask.cuda_mask(),
                                               duplicate_sources.cuda_mask(), layer_nb_edges);

    // get new edges on layer
    new_edges.reverse_build (&layer, edges_mask, _blocks);
    new_edges.tag_edges (_blocks, layer_connected_edges.layer_tag());
  }

  // sort edges by tag (=> edges are ordered by track)
  int *sorted_vector_positions;
  cudaMalloc (&sorted_vector_positions, layer_connected_edges.nb_edges() * sizeof(int));
  thrust_sort (sorted_vector_positions, layer_connected_edges.tags(), layer_connected_edges.nb_edges(), _blocks);

  layer.merge_edges (layer_connected_edges, new_edges, sorted_vector_positions, _blocks); // pb with sorted_vector size ?
  cudaFree (sorted_vector_positions);
//print_edge<<<1,1>>> (layer.source(), layer.target(), layer.weight(), layer.tags(), 44820, layer.nb_edges());
//print_edge<<<1,1>>> (layer.source(), layer.target(), layer.weight(), layer.tags(), 118239, layer.nb_edges());
print_edge_tag<<<1,1>>> (layer.source(), layer.target(), layer.weight(), layer.tags(), 64, layer.nb_edges());
print_edge_tag<<<1,1>>> (layer.source(), layer.target(), layer.weight(), layer.tags(), 108, layer.nb_edges());
}

//-------------------------------------------------------------------------------------------------
template <class T>
void CUDA_edge_layer_connector<T>::sort_v5 (CUDA_edges<T>& layer, //CUDA_edges<T>& last_edges,
                                            CUDA_edges<T>& next_layer, int layer_nb){
//-------------------------------------------------------------------------------------------------
  int layer_nb_edges = layer.nb_edges();
  dim3 grid_dim ((layer_nb_edges + _blocks - 1) / _blocks);
  CUDA_mask edges_mask (layer_nb_edges);
//std::cout << "layer = " << layer_nb << reset;
//std::cout << "/////////////////////////////////////////////////////--------------------------------------------------------\n";
//print_edge<<<1,1>>> (layer.source(), layer.target(), layer.weight(), layer.tags(), 332765, layer.nb_edges());
//print_edge<<<1,1>>> (layer.source(), layer.target(), layer.weight(), layer.tags(), 328198, layer.nb_edges());
cudaDeviceSynchronize();

  //--------------------------------------------------------------------
  // sort in layer the edges connected to next_layer edges and new edges
  //--------------------------------------------------------------------

  // create edges mask with 0 if layer.target is not in next_layer.source (new egde) and 1 if layer.target is in next_layer.source (existing edge)
  layer.connect_layer_edges (edges_mask, next_layer, _blocks);

  // untag duplicate sources
  CUDA_mask duplicate_sources (layer_nb_edges);
  layer.find_duplicate_sources (duplicate_sources, edges_mask, _blocks);

  // modify filter to avoid duplicate edges
//  mask_filter<<<grid_dim,_blocks>>> (edges_mask.cuda_mask(),
//                                     duplicate_sources.cuda_mask(), layer_nb_edges);

  // create layer_connected_edges from layer: separate layer edges (connected to next layer edges) and new edges
  CUDA_edges<T> layer_connected_edges (&layer, edges_mask, _blocks); // layer_connected_edges <-> (layer target edges == next layer source edges)

  // get edges layer tag from next_layer edges tag
  layer_connected_edges.copy_tags (next_layer, _blocks);

  // filter existing edges on cummulated weights
  layer_connected_edges.untag_duplicate_sources (_blocks);

  CUDA_edges<T> new_edges;

  if (layer_nb+2 >= _min_hits){ // layer 0 = 2 hits, layer 1 = 3 hits, layer 2 = 4 hits, layer 3 = 5 hits ...

    // kill new edges with duplicate sources (based on edges weigths - except if edge is linked to an edge of the next layer)
    mask_reverse_filter<<<grid_dim,_blocks>>> (edges_mask.cuda_mask(),
                                               duplicate_sources.cuda_mask(), layer_nb_edges);

    // filter on unallocated last edges remaining
//    layer.filter_last_edges (edges_mask, last_edges, _blocks); // new_edges not built yet !!!!!!!!!!!!!!!!!!!!!

    // get new edges on layer
    new_edges.reverse_build (&layer, edges_mask, _blocks);
    new_edges.tag_edges (_blocks, layer_connected_edges.layer_tag());
  }

  // sort edges by tag (=> edges are ordered by track)
  int *sorted_vector_positions;
  cudaMalloc (&sorted_vector_positions, layer_connected_edges.nb_edges() * sizeof(int));
  thrust_sort (sorted_vector_positions, layer_connected_edges.tags(), layer_connected_edges.nb_edges(), _blocks);

  layer.merge_edges (layer_connected_edges, new_edges, sorted_vector_positions, _blocks); // pb with sorted_vector size ?
//print_edge<<<1,1>>> (layer_connected_edges.source(), layer_connected_edges.target(), layer_connected_edges.weight(), layer_connected_edges.tags(), 164046, layer_connected_edges.nb_edges());
/*
print_edge<<<1,1>>> (layer.source(), layer.target(), layer.weight(), layer.tags(), 164046, layer.nb_edges());
print_edge_tag<<<1,1>>> (layer.source(), layer.target(), layer.weight(), layer.tags(), 28, layer.nb_edges());
print_edge_tag<<<1,1>>> (layer.source(), layer.target(), layer.weight(), layer.tags(), 80, layer.nb_edges());
std::cout << "--------------------------------------------------------\n";
*/

/*
print_edge<<<1,1>>> (layer.source(), layer.target(), layer.weight(), layer.tags(), 146490, layer.nb_edges());
print_edge<<<1,1>>> (layer.source(), layer.target(), layer.weight(), layer.tags(), 146589, layer.nb_edges());
print_edge<<<1,1>>> (layer.source(), layer.target(), layer.weight(), layer.tags(), 295600, layer.nb_edges());
print_edge<<<1,1>>> (layer.source(), layer.target(), layer.weight(), layer.tags(), 315867, layer.nb_edges());
print_edge_tag<<<1,1>>> (layer.source(), layer.target(), layer.weight(), layer.tags(), 116, layer.nb_edges());
print_edge_tag<<<1,1>>> (layer.source(), layer.target(), layer.weight(), layer.tags(), 204, layer.nb_edges());
//print_edge_tag<<<1,1>>> (layer.source(), layer.target(), layer.weight(), layer.tags(), 104, layer.nb_edges());
//print_edge_tag<<<1,1>>> (layer.source(), layer.target(), layer.weight(), layer.tags(), 193, layer.nb_edges());
std::cout << "--------------------------------------------------------\n";
*/
cudaDeviceSynchronize();

  cudaFree (sorted_vector_positions);
}

//---------------------------------------------------------------------------------------------------------------------
template <class T>
void CUDA_edge_layer_connector<T>::save (const std::string& filename){
//---------------------------------------------------------------------------------------------------------------------
  assert (_nb_layers);

  std::cout << cyan << "saving tracks in file " << filename << reset;
  (*_tracks).save (_min_hits, filename);
}
