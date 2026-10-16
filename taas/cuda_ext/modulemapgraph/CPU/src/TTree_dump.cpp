/***********************************************************************
 * Module Map Graph library
 * Track reconstruction for HEP
 * copyright © 2022,2023 ROUGIER Charline
 * copyright © 2022, 2023, 2024, 2025 COLLARD Christophe
 * copyright © 2025 VALLIER Alexis
 * Laboratoire des 2 Infinis - Toulouse - France (L2IT)
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 **********************************************************************/

#include "MMG/TTree_dump.hpp"

//---------------------------------------------------------------------------------------------------------------------
uint64_t make_particle_id(const int& subevent, const int& barcode, const uint& width){
//---------------------------------------------------------------------------------------------------------------------
  // Utility function to compute the particle ID from subevent number and barcode like in acorn
  std::ostringstream oss_barcode;
  oss_barcode << std::setw(width) << std::setfill('0') << barcode;
  uint64_t pid = std::stoull(std::to_string(subevent) + oss_barcode.str());
  return pid;
}

//---------------------------------------------------------------------------------------------------------------------
TTree_dump::TTree_dump(TTree* t){
//---------------------------------------------------------------------------------------------------------------------
    if(t==nullptr){
        std::cout<<"Error : trying to make a TTree_dump object with a null TTree!"<<std::endl;
        exit(1);
    }

    tree = t;

    event_number=0;

    // Particles
    nPartEVT=0;
    subevent=0;
    barcode=0;
    pt=0;
    eta=0;
    pdg_id=0;
    vx=0;
    vy=0;
    vz=0;

    // Space points
    nSP=0;
    SPindex=0;
    SPx=0;
    SPy=0;
    SPz=0;
    SPCL1_index=0;
    SPCL2_index=0;
    SPisOverlap=0;

    // Clusters
    nCL=0;
    CLindex=0;
    CLhardware=0;
    CLbarrel_endcap=0;
    CLeta_module=0;
    CLphi_module=0;
    CLmoduleID=0;
    CLparticleLink_eventIndex=0;
    CLparticleLink_barcode=0;

    Init();
}

//---------------------------------------------------------------------------------------------------------------------
TTree_dump::~TTree_dump(){
//---------------------------------------------------------------------------------------------------------------------
    delete subevent;
    delete barcode;
    delete pt;
    delete eta;
    delete pdg_id;
    delete vx;
    delete vy;
    delete vz;

    delete SPindex;
    delete SPx;
    delete SPy;
    delete SPz;
    delete SPCL1_index;
    delete SPCL2_index;
    delete SPisOverlap;

    delete CLindex;
    CLhardware->clear();
    delete CLbarrel_endcap;
    delete CLeta_module;
    delete CLphi_module;
    delete CLmoduleID;

    for(auto subevents : *CLparticleLink_eventIndex){
        subevents.clear();
    }
    delete CLparticleLink_eventIndex;
    
    for(auto barcodes : *CLparticleLink_barcode){
        barcodes.clear();
    }
    delete CLparticleLink_barcode;
}

//---------------------------------------------------------------------------------------------------------------------
void TTree_dump::GetEntry(Long64_t entry){
//---------------------------------------------------------------------------------------------------------------------
    if(tree){
        tree->GetEntry(entry);
    }
    else{
        std::cout<<"Warning : trying to get an entry in null tree of TTree_dump object"<<std::endl;
    }
}

//---------------------------------------------------------------------------------------------------------------------
void TTree_dump::Init(){
//---------------------------------------------------------------------------------------------------------------------
    if(!tree) return;

    // Activate only needed branches
    // -----------------------------
    tree->SetBranchStatus("*",false);

    tree->SetBranchStatus("event_number",true);

    // Particles
    tree->SetBranchStatus("nPartEVT",true);
    tree->SetBranchStatus("Part_event_number",true);
    tree->SetBranchStatus("Part_barcode",true);
    tree->SetBranchStatus("Part_pt",true);
    tree->SetBranchStatus("Part_eta",true);
    tree->SetBranchStatus("Part_pdg_id",true);
    tree->SetBranchStatus("Part_vx",true);
    tree->SetBranchStatus("Part_vy",true);
    tree->SetBranchStatus("Part_vz",true);

    // SP
    tree->SetBranchStatus("nSP",true);
    tree->SetBranchStatus("SPindex",true);
    tree->SetBranchStatus("SPx",true);
    tree->SetBranchStatus("SPy",true);
    tree->SetBranchStatus("SPz",true);
    tree->SetBranchStatus("SPCL1_index",true);
    tree->SetBranchStatus("SPCL2_index",true);
    tree->SetBranchStatus("SPisOverlap",true);

    // CL
    tree->SetBranchStatus("nCL",true);
    tree->SetBranchStatus("CLindex",true);
    tree->SetBranchStatus("CLhardware",true);
    tree->SetBranchStatus("CLbarrel_endcap",true);
    tree->SetBranchStatus("CLeta_module",true);
    tree->SetBranchStatus("CLphi_module",true);
    tree->SetBranchStatus("CLmoduleID",true);
    tree->SetBranchStatus("CLparticleLink_eventIndex",true);
    tree->SetBranchStatus("CLparticleLink_barcode",true);
    
    // Checking first the max of particles per event
    // ---------------------------------------------
    tree->SetBranchAddress("nPartEVT",&nPartEVT);
    tree->SetBranchAddress("nCL",&nCL);
    tree->SetBranchAddress("nSP",&nSP);
    int max_size_part=0;
    int max_size_cl=0;
    int max_size_sp=0;

    for(int i(0); i<tree->GetEntries(); ++i){
        tree->GetEntry(i);
        if(max_size_part<nPartEVT) max_size_part = nPartEVT;
        if(max_size_cl<nCL) max_size_cl = nCL;
        if(max_size_sp<nSP) max_size_sp = nSP;
    }

    // Finalize remaining branches setup
    // ---------------------------------
    subevent = new int[max_size_part];
    barcode = new int[max_size_part];
    pt = new float[max_size_part];
    eta = new float[max_size_part];
    pdg_id = new int[max_size_part];
    vx = new float[max_size_part];
    vy = new float[max_size_part];
    vz = new float[max_size_part];

    SPindex = new int[max_size_sp];
    SPx = new double[max_size_sp];
    SPy = new double[max_size_sp];
    SPz = new double[max_size_sp];
    SPCL1_index = new int[max_size_sp];
    SPCL2_index = new int[max_size_sp];
    SPisOverlap = new int[max_size_sp];

    CLindex = new int[max_size_cl];
    CLhardware = new std::vector<std::string>;
    CLbarrel_endcap = new int[max_size_cl];
    CLeta_module = new int[max_size_cl];
    CLphi_module = new int[max_size_cl];
    CLmoduleID = new ULong64_t[max_size_cl];
    CLparticleLink_eventIndex = new std::vector<std::vector<int>>;
    CLparticleLink_barcode = new std::vector<std::vector<int>>;
    
    tree->SetBranchAddress("event_number",&event_number);

    tree->SetBranchAddress("Part_event_number",subevent);
    tree->SetBranchAddress("Part_barcode",barcode);
    tree->SetBranchAddress("Part_pt",pt);
    tree->SetBranchAddress("Part_eta",eta);
    tree->SetBranchAddress("Part_pdg_id",pdg_id);
    tree->SetBranchAddress("Part_vx",vx);
    tree->SetBranchAddress("Part_vy",vy);
    tree->SetBranchAddress("Part_vz",vz);

    tree->SetBranchAddress("SPindex",SPindex);
    tree->SetBranchAddress("SPx",SPx);
    tree->SetBranchAddress("SPy",SPy);
    tree->SetBranchAddress("SPz",SPz);
    tree->SetBranchAddress("SPCL1_index",SPCL1_index);
    tree->SetBranchAddress("SPCL2_index",SPCL2_index);
    tree->SetBranchAddress("SPisOverlap",SPisOverlap);

    tree->SetBranchAddress("CLindex",CLindex);
    tree->SetBranchAddress("CLhardware",&CLhardware);
    tree->SetBranchAddress("CLbarrel_endcap",CLbarrel_endcap);
    tree->SetBranchAddress("CLeta_module",CLeta_module);
    tree->SetBranchAddress("CLphi_module",CLphi_module);
    tree->SetBranchAddress("CLmoduleID",CLmoduleID);
    tree->SetBranchAddress("CLparticleLink_eventIndex",&CLparticleLink_eventIndex);
    tree->SetBranchAddress("CLparticleLink_barcode",&CLparticleLink_barcode);
}