/***********************************************************************
 * Module Map Graph library
 * Track reconstruction for HEP
 * copyright © 2024,2025 COLLARD Christophe
 * copyright © 2025 PIGOU Maxime
 * Laboratoire des 2 Infinis - Toulouse - France (L2IT)
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 **********************************************************************/

#include "MMG/graph_creator.cuh"

//=====GPU functions for graph creator=======================================

/*
__global__ void print_value (int* vector, int pos)
{
  printf ("vector value at %d = %d \n", pos, vector[pos]);
}

__global__ void print (int* vector, int size)
{
  for (int i=0; i<size; i++)
    printf ("%d ", vector[i]);
}
*/

__global__ void check_vertex (int *input_vertices, int *output_vertices, int ivertex, int overtex, int nb_edges)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nb_edges) return;

  if ((input_vertices[i] == ivertex) && (output_vertices[i] == overtex))
    printf ("vertex found \n");
}


//=====Private methods for graph creator===============================================================================


//---------------------------------------------------------------------------------------------------------------------
template <class T, class module_map_type>
void CUDA_graph_creator<T,module_map_type>::get_program_options (const boost::program_options::variables_map& po_vm){
//---------------------------------------------------------------------------------------------------------------------
  _blocks = po_vm["gpu-nb-blocks"].as<int>();
  _max_blocks = po_vm["gpu-max-blocks"].as<int>();
  _device = po_vm["gpu-core"].as<int>();
  _events_dir = po_vm["input-dir"].as<std::string>() + "/";
  _output_dir = po_vm["output-dir"].as<std::string>();
  _true_graph = po_vm["give-true-graph"].as<bool>();
  _module_map_dir = po_vm["input-module-map"].template as<std::string>();
  _save_graph_graphml = po_vm["save-graph-on-disk-graphml"].as<bool>();
  _save_graph_npz = po_vm["save-graph-on-disk-npz"].as<bool>();
  _save_graph_pyg = po_vm["save-graph-on-disk-pyg"].as<bool>();
  _save_graph_csv = po_vm["save-graph-on-disk-csv"].as<bool>();
  _strip_hit_pair = po_vm["strip-hit-pair"].as<bool>();
  _extra_features = po_vm["extra-features"].as<bool>();
  _start_from = po_vm["restart-from"].as<int>();
  _end_at = po_vm["end-at"].as<int>();
  assert (_end_at >= _start_from);
  assert (_max_blocks);

  if (_strip_hit_pair) {
    _strip_module_DB = strip_module_DB<data_type2>(po_vm["input-strip-modules"].as<std::string>());
    _extra_features = true;
  }
}

//---------------------------------------------------------------------------------------------------------------------
template <class T, class module_map_type>
void CUDA_graph_creator<T,module_map_type>::read_event (const std::string& event_id, const std::string& hits_filename){
//---------------------------------------------------------------------------------------------------------------------
  // load event hits
  hits<T> hit_pts(_true_graph, _extra_features);
  TTree_hits<T> TThits (_true_graph, _extra_features);
  std::string file_extension = hits_filename.substr (hits_filename.length() - 4, hits_filename.length());
  if (file_extension == ".csv" ) {
    hit_pts.read_csv (_events_dir, hits_filename);
    TThits = hit_pts;
  }
  else if (file_extension == ".root")
    TThits.read (_events_dir + hits_filename);
  else
    throw std::invalid_argument("unknow extension for hit file: " + hits_filename);

  _cuda_hits[0].add_event (event_id, TThits, _cuda_module_map_doublet, _blocks, _max_blocks);
  _nb_events = 1;
}

//---------------------------------------------------------------------------------------------------------------------
template <class T, class module_map_type>
void CUDA_graph_creator<T,module_map_type>::read_events (std::ifstream& hits_file, std::ifstream& particles_file,
                                                         const std::string& pattern){
//---------------------------------------------------------------------------------------------------------------------
  std::string hits_filename, particles_filename;
  int count = 0;

  for (int i=0; !hits_file.eof() && !particles_file.eof() && i<_nb_events; i++) {
    hits_file >> hits_filename;
    particles_file >> particles_filename;
    if (hits_file.eof() || particles_file.eof() || (_end_at && (i<_start_from || i>_end_at))) continue;

    std::string event_id = extract_event_id (hits_filename, pattern);
    _event_id.push_back(event_id);
    std::cout << green << "event location " << _events_dir + hits_filename << reset;
    std::cout << "Event ID = " << event_id << std::endl;

  // load event hits
    hits<T> hit_pts(_true_graph, _extra_features);
    TTree_hits<T> TThits (_true_graph, _extra_features);
    std::string file_extension = hits_filename.substr (hits_filename.length() - 4, hits_filename.length());
    if (file_extension == ".csv" ) {
      hit_pts.read_csv (_events_dir, hits_filename);
      TThits = hit_pts;
    }
    else if (file_extension == ".root")
      TThits.read (_events_dir + hits_filename);
    else
      throw std::invalid_argument("unknow extension for hit file: " + hits_filename);

    _cuda_hits[count++].add_event (event_id, TThits, _cuda_module_map_doublet, _blocks, _max_blocks);
  }
  _nb_events = count;
}

//---------------------------------------------------------------------------------------------------------------------
template <class T, class module_map_type>
void CUDA_graph_creator<T,module_map_type>::initialize (int device){
//---------------------------------------------------------------------------------------------------------------------
  if (_module_map_dir.empty())
    throw std::invalid_argument ("Missing Module Map");

  module_map_triplet<module_map_type> module_map_triplet;
  module_map_triplet.read_TTree(_module_map_dir.c_str());
  if (!module_map_triplet)
    throw std::runtime_error("Cannot retrieve ModuleMap from " + _module_map_dir);

  initialize (device, module_map_triplet);
}

//---------------------------------------------------------------------------------------------------------------------
template <class T, class module_map_type>
void CUDA_graph_creator<T,module_map_type>::initialize (int device, module_map_triplet<module_map_type>& MMT){
//---------------------------------------------------------------------------------------------------------------------
  int nb_gpu = 0;
  cudaGetDeviceCount (&nb_gpu);
  assert (_device <= nb_gpu);
  std::cout << blue << "CUDA info: " << nb_gpu <<  " GPU found" << reset;
//  cudaStreamCreate(&_stream);
  _device = device;
  cudaSetDevice(device);
//  thrust::cuda::par.on(_stream);

  size_t f, t;
  cudaMemGetInfo(&f, &t);
  std::cout << blue << "CUDA infos: free memory " << f/pow(1024,3) << " Go / " << t/pow(1024,3) << "Go" << reset;

  cudaStream_t stream1, stream2;
  cudaStreamCreate(&stream1);
  cudaStreamCreate(&stream2);
  _cuda_module_map_doublet.init (MMT);
  _cuda_module_map_doublet.HostToDevice(stream1);
  _cuda_module_map_triplet.init (MMT);
  _cuda_module_map_triplet.HostToDevice(stream2);
  cudaStreamDestroy(stream1);
  cudaStreamDestroy(stream2);
  std::cout << "# of modules = " << MMT.module_map().size() << std::endl;

  clock_t start_alloc, end_alloc;
  start_alloc = clock ();
  _nb_doublets = _cuda_module_map_doublet.size();
  cudaMalloc (&_cuda_nb_edges, _nb_doublets * sizeof(int));
  cudaMalloc (&_cuda_edge_sum, (_nb_doublets + 1) * sizeof(int));
  _sort_by_size.reserve (_nb_doublets);
  end_alloc = clock ();
  std::cout << cyan << "cpu time for memory allocation: " << (double)(end_alloc - start_alloc) / CLOCKS_PER_SEC << reset;
}

//---------------------------------------------------------------------------------------------------------------------
template <class T, class module_map_type>
void CUDA_graph_creator<T,module_map_type>::initialize (module_map_triplet<module_map_type>& MMT,
                                                        int blocks, int max_blocks, int device){
//---------------------------------------------------------------------------------------------------------------------
  int nb_gpu = 0;
  cudaGetDeviceCount (&nb_gpu);
  assert (_device <= nb_gpu);
  assert (max_blocks);
  std::cout << blue << "CUDA info: " << nb_gpu <<  " GPU found" << reset;

  size_t f, t;
  cudaMemGetInfo(&f, &t);
  std::cout << blue << "CUDA infos: free memory " << f/pow(1024,3) << " Go / " << t/pow(1024,3) << "Go" << reset;

  _blocks = blocks;
  _max_blocks = max_blocks;
  _device = device;
  cudaSetDevice(device);
//  _stream = stream;
//  cudaStreamCreate(&_stream);
//  thrust::cuda::par.on(_stream);

  _cuda_module_map_doublet.init (MMT);
//  _cuda_module_map_doublet.HostToDevice();
  cudaStream_t _stream1, _stream2;
  cudaStreamCreate(&_stream1);
  cudaStreamCreate(&_stream2);
  _cuda_module_map_doublet.HostToDevice(_stream1);
  _cuda_module_map_triplet.init (MMT);
  _cuda_module_map_triplet.HostToDevice(_stream2);
  cudaStreamDestroy(_stream1);
  cudaStreamDestroy(_stream2);
  std::cout << "# of modules = " << MMT.module_map().size() << std::endl;

  clock_t start_alloc, end_alloc;
  start_alloc = clock ();
  _nb_doublets = _cuda_module_map_doublet.size();
  cudaMalloc (&_cuda_nb_edges, _nb_doublets * sizeof(int));
  cudaMalloc (&_cuda_edge_sum, (_nb_doublets + 1) * sizeof(int));
  _sort_by_size.reserve (_nb_doublets);
  end_alloc = clock ();
  std::cout << cyan << "cpu time for memory allocation: " << (double)(end_alloc - start_alloc) / CLOCKS_PER_SEC << reset;
}

// Count the number of edges while parallelizing over the source hits
//---------------------------------------------------------------------------------------------------------------------
template <class T, class module_map_type>
int CUDA_graph_creator<T,module_map_type>::count_doublets (int *cuda_edge_sum_per_src_hit,
                      const int *cuda_nb_src_hits_per_doublet, const int& sum_nb_src_hits_per_doublet,
                      CUDA_hits<T>& cuda_hits){
//---------------------------------------------------------------------------------------------------------------------
  dim3 grid_dim_shpd = ((sum_nb_src_hits_per_doublet + _blocks - 1) / _blocks);
  CUDA_module_map_doublet<T>* MM = &_cuda_module_map_doublet;

  count_doublet_edges<<<grid_dim_shpd,_blocks>>> (cuda_edge_sum_per_src_hit,
                     _nb_doublets, cuda_nb_src_hits_per_doublet,
                     _cuda_module_map_doublet.cuda_module1(), _cuda_module_map_doublet.cuda_module2(),
                     cuda_hits.cuda_R(), cuda_hits.cuda_z(), cuda_hits.cuda_eta(), cuda_hits.cuda_phi(),
                     (*MM).cuda_z0_min(), (*MM).cuda_deta_min(), (*MM).cuda_phi_slope_min(), (*MM).cuda_dphi_min(),
                     (*MM).cuda_z0_max(), (*MM).cuda_deta_max(), (*MM).cuda_phi_slope_max(), (*MM).cuda_dphi_max(),
                     cuda_hits.cuda_hit_indice(), std::numbers::pi_v<T>, epsilon<T>(), sum_nb_src_hits_per_doublet);

  // Extract the number of edges and copy to host
  int nb_doublet_edges = thrust_scan (cuda_edge_sum_per_src_hit, sum_nb_src_hits_per_doublet);

  return nb_doublet_edges;
}

//---------------------------------------------------------------------------------------------------------------------
template <class T, class module_map_type>
void CUDA_graph_creator<T,module_map_type>::build_doublets (int *cuda_reduced_M1_hits, int *cuda_reduced_M2_hits,
                      const int *cuda_nb_src_hits_per_doublet, int *cuda_edge_sum_per_src_hit, CUDA_hits<T>& cuda_hits,
                      int sum_nb_src_hits_per_doublet){
//---------------------------------------------------------------------------------------------------------------------
  dim3 grid_dim_shpd = ((sum_nb_src_hits_per_doublet + _blocks - 1) / _blocks);
  CUDA_module_map_doublet<T>* MM = &_cuda_module_map_doublet;

  build_doublet_edges<<<grid_dim_shpd,_blocks>>> (cuda_reduced_M1_hits, cuda_reduced_M2_hits,
                    _nb_doublets, cuda_nb_src_hits_per_doublet, cuda_hits.cuda_hit_indice(), cuda_edge_sum_per_src_hit,
                    _cuda_module_map_doublet.cuda_module1(), _cuda_module_map_doublet.cuda_module2(),
                    cuda_hits.cuda_R(), cuda_hits.cuda_z(), cuda_hits.cuda_eta(), cuda_hits.cuda_phi(),
                    (*MM).cuda_z0_min(), (*MM).cuda_deta_min(), (*MM).cuda_phi_slope_min(), (*MM).cuda_dphi_min(),
                    (*MM).cuda_z0_max(), (*MM).cuda_deta_max(), (*MM).cuda_phi_slope_max(), (*MM).cuda_dphi_max(),
                    std::numbers::pi_v<T>, epsilon<T>(), sum_nb_src_hits_per_doublet);

  // Build the prefix sum of the number of edges for the doublets
  dim3 grid_dim ((_nb_doublets + _blocks - 1) / _blocks);
  doublet_edge_sum<<<grid_dim,_blocks>>> (_cuda_edge_sum,
                                          cuda_nb_src_hits_per_doublet, cuda_edge_sum_per_src_hit, _nb_doublets);
}

//---------------------------------------------------------------------------------------------------------------------
template <class T, class module_map_type>
void CUDA_graph_creator<T,module_map_type>::build_triplets (CUDA_hits<T>& cuda_hits, CUDA_mask& cuda_edge_tag,
                       int nb_triplets, int *cuda_src_hits_per_triplet,
                       T *cuda_z0, T *cuda_phi_slope, T *cuda_deta, T *cuda_dphi,
                       int *cuda_reduced_M1_hits, int *cuda_reduced_M2_hits, int nb_src_hits_per_triplet_sum){
//---------------------------------------------------------------------------------------------------------------------
  dim3 grid_dim_shpt = ((nb_src_hits_per_triplet_sum + _blocks - 1) / _blocks);

  CUDA_module_map_doublet<T> *M12 = &_cuda_module_map_triplet.module12();
  CUDA_module_map_doublet<T> *M23 = &_cuda_module_map_triplet.module23();

  triplet_cuts<T><<<grid_dim_shpt,_blocks>>> (cuda_hits.cuda_vertices(), cuda_edge_tag.cuda_mask(),
        nb_triplets, cuda_src_hits_per_triplet,
        _cuda_module_map_triplet.cuda_module12_map(), _cuda_module_map_triplet.cuda_module23_map(),
        cuda_hits.cuda_x(), cuda_hits.cuda_y(), cuda_hits.cuda_z(), cuda_hits.cuda_R(),
        cuda_z0, cuda_phi_slope, cuda_deta, cuda_dphi,
        M12->cuda_z0_min(), M12->cuda_phi_slope_min(), M12->cuda_deta_min(), M12->cuda_dphi_min(),
        M12->cuda_z0_max(), M12->cuda_phi_slope_max(), M12->cuda_deta_max(), M12->cuda_dphi_max(),
        M23->cuda_z0_min(), M23->cuda_phi_slope_min(), M23->cuda_deta_min(), M23->cuda_dphi_min(),
        M23->cuda_z0_max(), M23->cuda_phi_slope_max(), M23->cuda_deta_max(), M23->cuda_dphi_max(),
        _cuda_module_map_triplet.cuda_diff_dydx_min(), _cuda_module_map_triplet.cuda_diff_dydx_max(),
        _cuda_module_map_triplet.cuda_diff_dzdr_min(), _cuda_module_map_triplet.cuda_diff_dzdr_max(),
        cuda_reduced_M1_hits, cuda_reduced_M2_hits,
        _cuda_edge_sum, std::numbers::pi_v<T>, epsilon<T>(), nb_src_hits_per_triplet_sum);
}

//=====Public methods for graph creator================================================================================


//---------------------------------------------------------------------------------------------------------------------
template <class T, class module_map_type>
CUDA_graph_creator<T,module_map_type>::CUDA_graph_creator (const boost::program_options::variables_map &po_vm){
//---------------------------------------------------------------------------------------------------------------------
  get_program_options (po_vm);
  initialize (po_vm["gpu-core"].as<int>());

  std::string event_filename_pattern = po_vm["input-filename-pattern"].as<std::string>();
  // log all hits events filenames
  std::string log_hits_filenames = "events-hits.txt";
  std::string cmd_hits = "ls " + _events_dir + "|grep " + event_filename_pattern + " |grep hits.* > " + log_hits_filenames;
  system(cmd_hits.c_str());
  cmd_hits = "ls " + _events_dir + "|grep " +  event_filename_pattern + " |grep truth.* >> " + log_hits_filenames;
  system(cmd_hits.c_str());

  // log all particles events filenames
  std::string log_particles_filenames = "events-particles.txt";
  std::string cmd_particles = "ls " + _events_dir + "|grep event |grep particles.* > " + log_particles_filenames;
  system (cmd_particles.c_str());

  // open files containing events filenames
  std::ifstream h_file(log_hits_filenames);
  if (h_file.fail())
    throw std::invalid_argument("Cannot open file " + log_hits_filenames);
  std::ifstream p_file(log_particles_filenames);
  if (p_file.fail())
    throw std::invalid_argument("Cannot open file " + log_particles_filenames);

  std::string hits_filename, particles_filename;

  std::string log_nb_events = "events_nb.txt";
  std::string cmd_nb_events = "wc -l " + log_hits_filenames + " > " + log_nb_events;
  system (cmd_nb_events.c_str());
  std::ifstream e_file (log_nb_events);
  e_file >> _nb_events;
  std::cout << red << "nb events = " << _nb_events << reset;
  e_file.close();

  clock_t start_loop, end_loop;
  start_loop = clock();
  _cuda_hits = new CUDA_hits<T> [_nb_events];
  read_events (h_file, p_file, event_filename_pattern);
  end_loop = clock();
  std::cout << cyan << "cpu time for " << _nb_events << " events: " << (double)(end_loop - start_loop) / CLOCKS_PER_SEC << reset;

  h_file.close();
  p_file.close();
}

// constructor for multi-threads run
//---------------------------------------------------------------------------------------------------------------------
template <class T, class module_map_type>
CUDA_graph_creator<T,module_map_type>::CUDA_graph_creator (const boost::program_options::variables_map &po_vm,
                                                           int device, module_map_triplet<module_map_type>& MMT,
                                                           std::string event_id, const std::string& hits_filename){
//---------------------------------------------------------------------------------------------------------------------
  get_program_options (po_vm);
  initialize (device, MMT);
//  _cuda_hits = new CUDA_hits<T>;
  read_event (event_id, hits_filename);
}

// constructor for ACORN
//---------------------------------------------------------------------------------------------------------------------
template <class T, class module_map_type>
CUDA_graph_creator<T,module_map_type>::CUDA_graph_creator (const boost::program_options::variables_map &po_vm,
                                                           bool nofiles){
//---------------------------------------------------------------------------------------------------------------------
  get_program_options (po_vm);
  initialize (po_vm["gpu-core"].as<int>());
}

// constructor for ACTS
//---------------------------------------------------------------------------------------------------------------------
template <class T, class module_map_type>
CUDA_graph_creator<T,module_map_type>::CUDA_graph_creator (int blocks, int device, const std::string &module_map_dir){
//---------------------------------------------------------------------------------------------------------------------
  _blocks = blocks;
  _max_blocks = blocks;
  _device = device;
  _module_map_dir = module_map_dir;

  _true_graph = false;
  _save_graph_graphml = false;
  _save_graph_npz = false;
  _save_graph_pyg = false;
  _save_graph_csv = false;
  _strip_hit_pair = false;
  _extra_features = false;

  initialize (device);
}

//---------------------------------------------------------------------------------------------------------------------
template <class T, class module_map_type>
CUDA_graph_creator<T,module_map_type>::~CUDA_graph_creator (){
//---------------------------------------------------------------------------------------------------------------------
  cudaFree (_cuda_edge_sum);
  cudaFree (_cuda_nb_edges);
//  cudaStreamDestroy(_stream);
}

// constructor for multi-threads run
//---------------------------------------------------------------------------------------------------------------------
template <class T, class module_map_type>
void CUDA_graph_creator<T,module_map_type>::init (const boost::program_options::variables_map &po_vm,
                                                  module_map_triplet<module_map_type>& MMT,
                                                  std::string event_id, const std::string& hits_filename,
                                                  int blocks, int max_blocks, int device, cudaStream_t& stream){
//---------------------------------------------------------------------------------------------------------------------
  get_program_options (po_vm);
  initialize (MMT, blocks, max_blocks, device);
  _cuda_hits = new CUDA_hits<T>;
  read_event (event_id, hits_filename);
//  _stream = stream;
//  cudaStreamCreate(&_stream);
}

//---------------------------------------------------------------------------------------------------------------------
template <class T, class module_map_type>
void CUDA_graph_creator<T,module_map_type>::build (CUDA_nodes<T>& nodes, CUDA_edges<T>& edges, CUDA_hits<T>& cuda_hits)
//---------------------------------------------------------------------------------------------------------------------
{
#ifdef MMG_ENABLE_TIMING
  clock_t start_timer, end_timer;
  clock_t start, end;
  clock_t start_deco, end_deco;
  clock_t start_count_edges, end_count_edges;
  clock_t start_doublets, end_doublets;
  clock_t start_sum, end_sum;
  clock_t start_geocuts, end_geocuts;
  clock_t start_triplets, end_triplets;
  clock_t start_reduction, end_reduction;
  clock_t start_free, end_free;

  cudaDeviceSynchronize();
  start = clock();
  std::chrono::system_clock::time_point start_real = std::chrono::system_clock::now();
#endif

  _event_id.push_back(cuda_hits.event_id());

//  cudaStreamSynchronize (_stream);
#ifdef MMG_ENABLE_TIMING
  cudaDeviceSynchronize ();
  start_timer = start_deco = clock ();
#endif

  // ------------------------------
  // compute hits decoration values
  // ------------------------------

  cuda_hits.decorations();
//  cuda_hits.decorations (_device, _stream);

//  cudaStreamSynchronize (_stream);
#ifdef MMG_ENABLE_TIMING
  cudaDeviceSynchronize ();
  end_deco = start_count_edges = clock ();
#endif

  // ----------------------------------------
  // doublet cuts in the loop unrolling style
  // ----------------------------------------

  // Count the number of source-hits per doublet

  dim3 grid_dim ((_nb_doublets + _blocks - 1) / _blocks);
  int *cuda_nb_src_hits_per_doublet = nullptr;
  cudaMalloc (&cuda_nb_src_hits_per_doublet, (_nb_doublets + 1) * sizeof(int));

  count_source_hits_per_doublet<<<grid_dim,_blocks>>> (cuda_nb_src_hits_per_doublet,
                               _cuda_module_map_doublet.cuda_module1(), cuda_hits.cuda_hit_indice(), _nb_doublets+1);
//  count_source_hits_per_doublet<<<grid_dim,_blocks,_device,_stream>>> (cuda_nb_src_hits_per_doublet,
//                               _cuda_module_map_doublet.cuda_module1(), cuda_hits.cuda_hit_indice(), _nb_doublets+1);
//  cudaStreamSynchronize(_stream);

  int sum_nb_src_hits_per_doublet = thrust_scan (cuda_nb_src_hits_per_doublet, _nb_doublets);

  if (!sum_nb_src_hits_per_doublet) {
    std::cout << "no edges, return" << std::endl;
    return;
  }

  // Count the number of edges while parallelizing over the source hits
  int *cuda_edge_sum_per_src_hit;
  cudaMalloc (&cuda_edge_sum_per_src_hit, (sum_nb_src_hits_per_doublet + 1) * sizeof(int));
  int nb_doublet_edges = count_doublets (cuda_edge_sum_per_src_hit,
                                         cuda_nb_src_hits_per_doublet, sum_nb_src_hits_per_doublet, cuda_hits);

#ifdef MMG_ENABLE_TIMING
  cudaDeviceSynchronize();
  end_count_edges = start_doublets = clock();
#endif

  // Build the doublet edges

  int *cuda_reduced_M1_hits{}, *cuda_reduced_M2_hits{};
  cudaMalloc (&cuda_reduced_M1_hits, nb_doublet_edges * sizeof(int));
  cudaMalloc (&cuda_reduced_M2_hits, nb_doublet_edges * sizeof(int));

  build_doublets (cuda_reduced_M1_hits, cuda_reduced_M2_hits,
                  cuda_nb_src_hits_per_doublet, cuda_edge_sum_per_src_hit, cuda_hits, sum_nb_src_hits_per_doublet);

  cudaFree (cuda_edge_sum_per_src_hit);
  cudaFree (cuda_nb_src_hits_per_doublet);

#ifdef MMG_ENABLE_TIMING
  cudaDeviceSynchronize ();
  end_doublets = start_sum = clock();
#endif

#ifdef MMG_ENABLE_TIMING
  cudaDeviceSynchronize();
  end_sum = start_geocuts = clock();
#endif

  // -----------------------------
  // build doublets geometric cuts
  // -----------------------------

  T *cuda_z0, *cuda_phi_slope, *cuda_deta, *cuda_dphi;
  cudaMalloc (&cuda_z0, nb_doublet_edges * sizeof(T));
  cudaMalloc (&cuda_phi_slope, nb_doublet_edges * sizeof(T));
  cudaMalloc (&cuda_deta, nb_doublet_edges * sizeof(T));
  cudaMalloc (&cuda_dphi, nb_doublet_edges * sizeof(T));
  grid_dim = (nb_doublet_edges + _blocks - 1) / _blocks;
  hits_geometric_cuts<<<grid_dim,_blocks>>> (cuda_z0, cuda_phi_slope, cuda_deta, cuda_dphi,
                     cuda_reduced_M1_hits, cuda_reduced_M2_hits,
                     cuda_hits.cuda_R(), cuda_hits.cuda_z(), cuda_hits.cuda_eta(), cuda_hits.cuda_phi(),
                     std::numbers::pi_v<T>, epsilon<T>(), nb_doublet_edges);

#ifdef MMG_ENABLE_TIMING
  cudaDeviceSynchronize();
  end_geocuts = start_triplets = clock();
#endif

  // -------------------------
  // loop over module triplets
  // -------------------------

  int nb_triplets = _cuda_module_map_triplet.size();
  grid_dim = (nb_triplets + _blocks - 1) / _blocks;

   // Allocate memory for the number of hits per triplet
  int *cuda_src_hits_per_triplet = nullptr;
  cudaMalloc(&cuda_src_hits_per_triplet, (nb_triplets + 1) * sizeof(int));

  count_triplet_hits<<<grid_dim,_blocks>>> (cuda_src_hits_per_triplet,
                    _cuda_module_map_triplet.cuda_module12_map(), _cuda_module_map_triplet.cuda_module23_map(),
                    _cuda_edge_sum, nb_triplets);

  // Perform prefix sum to get the offset for each triplet
  int nb_src_hits_per_triplet_sum = thrust_scan (cuda_src_hits_per_triplet, nb_triplets);
  if (nb_src_hits_per_triplet_sum == 0) {
    std::cout << "no edges, return" << std::endl;
    return;
  }

  CUDA_mask cuda_edge_tag (nb_doublet_edges);
  build_triplets (cuda_hits, cuda_edge_tag,
                  nb_triplets, cuda_src_hits_per_triplet,
                  cuda_z0, cuda_phi_slope, cuda_deta, cuda_dphi,
                  cuda_reduced_M1_hits, cuda_reduced_M2_hits, nb_src_hits_per_triplet_sum);

  cudaFree (cuda_src_hits_per_triplet);

#ifdef MMG_ENABLE_TIMING
  cudaDeviceSynchronize();
  end_triplets = start_reduction = clock();
#endif

  //--------------------------
  // nodes and edges reduction
  //--------------------------

  nodes.reduce (cuda_hits, _blocks);
  edges.reduce (cuda_reduced_M1_hits, cuda_reduced_M2_hits, cuda_deta, cuda_dphi, cuda_phi_slope, cuda_edge_tag, cuda_hits, _blocks);
  edges.features (cuda_hits, _blocks);

#ifdef MMG_ENABLE_TIMING
  cudaDeviceSynchronize();
  end_reduction = clock();
#endif

  // get memory info
  size_t free_memory, total_memory;
  cudaMemGetInfo (&free_memory, &total_memory);
  GPU_ERROR_CHECK (cudaGetLastError());

  //-------------------
  // free memory on GPU
  //-------------------

#ifdef MMG_ENABLE_TIMING
  cudaDeviceSynchronize();
  start_free = clock();
#endif

  cudaFree (cuda_dphi);
  cudaFree (cuda_deta);
  cudaFree (cuda_phi_slope);
  cudaFree (cuda_z0);
  cudaFree (cuda_reduced_M2_hits);
  cudaFree (cuda_reduced_M1_hits);

#ifdef MMG_ENABLE_TIMING
  cudaDeviceSynchronize();
  end_free = end = end_timer = clock();
  std::chrono::system_clock::time_point end_real = std::chrono::system_clock::now();

  std::cout << "nb edges after MD cuts = " << nb_doublet_edges << std::endl;
  std::cout << "nb edges after MT cuts = " << edges.nb_edges() << std::endl;

  std::cout << cyan << "cpu time for hits decorations: " << (double)(end_deco - start_deco) / CLOCKS_PER_SEC << reset;
  std::cout << cyan << "cpu time to calculate the nb of edges: " << (double)(end_count_edges - start_count_edges) / CLOCKS_PER_SEC << reset;
  std::cout << cyan << "cpu time for doublets selection: " << (double)(end_doublets - start_doublets) / CLOCKS_PER_SEC << reset;
  std::cout << cyan << "cpu time for sum edges: " << (double)(end_sum - start_sum) / CLOCKS_PER_SEC << reset;
  std::cout << cyan << "cpu time for geometric cuts (doublets only): " << (double)(end_geocuts - start_geocuts) / CLOCKS_PER_SEC << reset;
  std::cout << cyan << "cpu time for geometric cuts (triplets only): " << (double)(end_triplets - start_triplets) / CLOCKS_PER_SEC << reset;
  std::cout << cyan << "cpu time for reduction: " << (double)(end_reduction - start_reduction) / CLOCKS_PER_SEC << reset;
  std::cout << cyan << "cpu time to free memory: " << (double)(end_free - start_free) / CLOCKS_PER_SEC << reset;
  std::cout << magenta << "cpu time for global selection: " << (double)(end - start) / CLOCKS_PER_SEC << reset;
  std::cout << red << "cpu time for graph builder: " << (double)(end - start_timer) / CLOCKS_PER_SEC << reset;
  std::cout << blue << "CUDA infos: used memory " << (total_memory - free_memory)/pow(1024,3) << " Go / " << total_memory/pow(1024,3) << "Go" << reset;

  std::ofstream outfile;
  outfile.open("GB.excel-data.txt", std::ios_base::app);
//  outfile.open("/scratch_ssd/chcollar/Tracking/excel-data.txt", std::ios_base::app);

  outfile << (double)(end_timer - start_timer) / CLOCKS_PER_SEC << ","
          << std::chrono::duration_cast<std::chrono::microseconds>(end_real - start_real).count() * 0.001 << ","
          << edges.nb_edges() << ","
          << nodes.nb_nodes() << ","
          << (double)(end_deco - start_deco) / CLOCKS_PER_SEC << ","
          << (double)(end_count_edges - start_count_edges) / CLOCKS_PER_SEC << ","
          << (double)(end_doublets - start_doublets) / CLOCKS_PER_SEC << ","
          << (double)(end_sum - start_sum) / CLOCKS_PER_SEC << ","
          << (double)(end_geocuts - start_geocuts) / CLOCKS_PER_SEC << ","
          << (double)(end_triplets - start_triplets) / CLOCKS_PER_SEC << ","
          << (double)(end_reduction - start_reduction) / CLOCKS_PER_SEC << ","
          << (double)(end_free - start_free) / CLOCKS_PER_SEC << ","
          << (total_memory - free_memory)/pow(1024,3)
          << std::endl;
  outfile.close();
#endif

  GPU_ERROR_CHECK (cudaGetLastError());
}

// ACORN and ACTS build function
//---------------------------------------------------------------------------------------------------------------------
template <class T, class module_map_type>
void CUDA_graph_creator<T,module_map_type>::build (CUDA_nodes<T>& nodes, CUDA_edges<T>& edges,
                       std::string eventID, uint64_t* hitID, uint64_t* moduleID, T* x, T* y, T* z, int size){
//---------------------------------------------------------------------------------------------------------------------
  cudaDeviceSynchronize();
  clock_t start = clock();
  CUDA_hits<T> cuda_hits (_cuda_module_map_doublet, hitID, moduleID, x, y, z, size, _blocks, _max_blocks);
  cudaDeviceSynchronize();
  clock_t end = clock();
//  std::cout << "pre-processing hit cpu time = " << (double)(end - start) / CLOCKS_PER_SEC << std::endl;

  build (nodes, edges, cuda_hits);
}

//---------------------------------------------------------------------------------------------------------------------
template <class T, class module_map_type>
void CUDA_graph_creator<T,module_map_type>::build (CUDA_nodes<T>& nodes, CUDA_edges<T>& edges, int event){
//---------------------------------------------------------------------------------------------------------------------
  assert (event < _nb_events);
  build (nodes, edges, _cuda_hits[event]);
}

// multi threads build
//---------------------------------------------------------------------------------------------------------------------
template <class T, class module_map_type>
void CUDA_graph_creator<T,module_map_type>::multithreads_build (CUDA_nodes<T>& nodes, CUDA_edges<T>& edges,
                                                                cudaStream_t& stream){
//---------------------------------------------------------------------------------------------------------------------
// read event here
//  read_event (event_id, hits_filename);
// create cuda_hit here
  build (nodes, edges, 0);
//  cudaStreamDestroy(_stream);
}

//---------------------------------------------------------------------------------------------------------------------
template <class T, class module_map_type>
std::string& CUDA_graph_creator<T,module_map_type>::get_event_id (int event){
//---------------------------------------------------------------------------------------------------------------------
  assert (event>0 && event<=_nb_events);

  return _event_id[event-1];
}

//---------------------------------------------------------------------------------------------------------------------
template <class T, class module_map_type>
void CUDA_graph_creator<T,module_map_type>::save (CUDA_nodes<T>* nodes, CUDA_edges<T>* edges){
//---------------------------------------------------------------------------------------------------------------------
  for (int event=0; event<_nb_events && event<_event_id.size(); event++)
    if (nodes[event].nb_nodes()) {
      int nb_nodes = nodes[event].nb_nodes();
      int nb_edges = edges[event].nb_edges();

      // --------------------
      // copy data GPU -> CPU
      // --------------------

      std::vector<uint64_t> hitsID (nb_nodes);
      std::vector<T> R(nb_nodes), z(nb_nodes), eta(nb_nodes), phi(nb_nodes);
      cudaMemcpy (hitsID.data(), nodes[event].cuda_hitID(), nb_nodes * sizeof(uint64_t), cudaMemcpyDeviceToHost);
      cudaMemcpy (R.data(), nodes[event].cuda_R(), nb_nodes * sizeof(T), cudaMemcpyDeviceToHost);
      cudaMemcpy (z.data(), nodes[event].cuda_z(), nb_nodes * sizeof(T), cudaMemcpyDeviceToHost);
      cudaMemcpy (eta.data(), nodes[event].cuda_eta(), nb_nodes * sizeof(T), cudaMemcpyDeviceToHost);
      cudaMemcpy (phi.data(), nodes[event].cuda_phi(), nb_nodes * sizeof(T), cudaMemcpyDeviceToHost);

      std::vector<int> M1_hits (nb_edges), M2_hits (nb_edges);

      std::vector<T> dR(nb_edges), dz(nb_edges), deta(nb_edges), dphi(nb_edges), phi_slope(nb_edges), r_phi_slope(nb_edges);

      cudaMemcpy (M1_hits.data(), edges[event].source(), nb_edges * sizeof(int), cudaMemcpyDeviceToHost);
      cudaMemcpy (M2_hits.data(), edges[event].target(), nb_edges * sizeof(int), cudaMemcpyDeviceToHost);
      cudaMemcpy (dR.data(), edges[event].dR(), nb_edges * sizeof(T), cudaMemcpyDeviceToHost);
      cudaMemcpy (dz.data(), edges[event].dz(), nb_edges * sizeof(T), cudaMemcpyDeviceToHost);
      cudaMemcpy (deta.data(), edges[event].deta(), nb_edges * sizeof(T), cudaMemcpyDeviceToHost);
      cudaMemcpy (dphi.data(), edges[event].dphi(), nb_edges * sizeof(T), cudaMemcpyDeviceToHost);
      cudaMemcpy (phi_slope.data(), edges[event].phi_slope(), nb_edges * sizeof(T), cudaMemcpyDeviceToHost);
      cudaMemcpy (r_phi_slope.data(), edges[event].r_phi_slope(), nb_edges * sizeof(T), cudaMemcpyDeviceToHost);

      cudaDeviceSynchronize();

      std::cout << green << "nb nodes = " << nb_nodes << reset;

      graph<T> G;

      for (int i=0; i<nb_nodes; i++)
        vertex_type<T> v1 = G.add_vertex (hitsID[i], R[i], z[i], eta[i], phi[i]);

      for (int i=0; i<nb_edges; i++) {
        edge<T> edg (deta[i], dphi[i], dR[i], dz[i], phi_slope[i], r_phi_slope[i]);
        G.add_edge (M1_hits[i], M2_hits[i], edg);
      }

      if (_save_graph_graphml) {
        std::cout << "writing output in directory " << _output_dir << std::endl;
        std::cout << "event id = " << _event_id[event] << std::endl;
        G.save (_event_id[event], _output_dir, _extra_features);
//        if (_true_graph) G_true.save (event_id, _output_dir);
      }

#ifdef MMG_WITH_NUMPY_EXPORT
      if (_save_graph_npz) {
      G.save_npz (_event_id[event], _output_dir, _extra_features);
//      if (_true_graph) G_true.save_npz (event_id, _output_dir);
      }
#endif

/*
      if (_save_graph_pyg) {
        G.save_pyg (event_id, _output_dir, _extra_features);
        //if (_true_graph) G_true.save_pyg (event_id, _output_dir);
      }
  */

      if (_save_graph_csv) {
        G.save_csv (_event_id[event], _output_dir, _extra_features);
    //        if (_true_graph) G_true.save (event_id, _output_dir);
      }
    }
}

//---------------------------------------------------------------------------------------------------------------------
template <class T, class module_map_type>
void CUDA_graph_creator<T,module_map_type>::save (CUDA_nodes<T>& nodes, CUDA_edges<T>& edges, std::string eventID){
//---------------------------------------------------------------------------------------------------------------------
  if (nodes.nb_nodes()) {
    int nb_nodes = nodes.nb_nodes();
    int nb_edges = edges.nb_edges();

    // --------------------
    // copy data GPU -> CPU
    // --------------------

    std::vector<uint64_t> hitsID (nb_nodes);
    std::vector<T> R(nb_nodes), z(nb_nodes), eta(nb_nodes), phi(nb_nodes);
    cudaMemcpy (hitsID.data(), nodes.cuda_hitID(), nb_nodes * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy (R.data(), nodes.cuda_R(), nb_nodes * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy (z.data(), nodes.cuda_z(), nb_nodes * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy (eta.data(), nodes.cuda_eta(), nb_nodes * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy (phi.data(), nodes.cuda_phi(), nb_nodes * sizeof(T), cudaMemcpyDeviceToHost);

    std::vector<int> M1_hits (nb_edges), M2_hits (nb_edges);

    std::vector<T> dR(nb_edges), dz(nb_edges), deta(nb_edges), dphi(nb_edges), phi_slope(nb_edges), r_phi_slope(nb_edges);

    cudaMemcpy (M1_hits.data(), edges.source(), nb_edges * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy (M2_hits.data(), edges.target(), nb_edges * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy (dR.data(), edges.dR(), nb_edges * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy (dz.data(), edges.dz(), nb_edges * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy (deta.data(), edges.deta(), nb_edges * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy (dphi.data(), edges.dphi(), nb_edges * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy (phi_slope.data(), edges.phi_slope(), nb_edges * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy (r_phi_slope.data(), edges.r_phi_slope(), nb_edges * sizeof(T), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    std::cout << green << "nb nodes = " << nb_nodes << reset;

    graph<T> G;

    for (int i=0; i<nb_nodes; i++)
      vertex_type<T> v1 = G.add_vertex (hitsID[i], R[i], z[i], eta[i], phi[i]);

    for (int i=0; i<nb_edges; i++) {
      edge<T> edg (deta[i], dphi[i], dR[i], dz[i], phi_slope[i], r_phi_slope[i]);
      G.add_edge (M1_hits[i], M2_hits[i], edg);
    }

    if (_save_graph_graphml) {
      std::cout << "writing output in directory " << _output_dir << std::endl;
      std::cout << "event id = " << eventID << std::endl;
      G.save (eventID, _output_dir, _extra_features);
//        if (_true_graph) G_true.save (event_id, _output_dir);
    }

#ifdef MMG_WITH_NUMPY_EXPORT
      if (_save_graph_npz) {
      G.save_npz (eventID, _output_dir, _extra_features);
//      if (_true_graph) G_true.save_npz (event_id, _output_dir);
      }
#endif

/*
      if (_save_graph_pyg) {
        G.save_pyg (event_id, _output_dir, _extra_features);
        //if (_true_graph) G_true.save_pyg (event_id, _output_dir);
      }
  */

      if (_save_graph_csv) {
        G.save_csv (eventID, _output_dir, _extra_features);
    //        if (_true_graph) G_true.save (event_id, _output_dir);
      }
    }
}
