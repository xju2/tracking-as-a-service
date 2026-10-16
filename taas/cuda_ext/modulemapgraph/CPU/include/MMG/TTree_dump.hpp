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

#include <iostream>
#include <iomanip>
#include <vector>
#include <TTree.h>

uint64_t make_particle_id(const int& subevent, const int& barcode, const uint& width=7);

//=====================================================================================================================
class TTree_dump{
//=====================================================================================================================
    public:
        TTree *tree;

        ULong64_t event_number;

        // Particles
        int nPartEVT;
        int* subevent;
        int* barcode;
        float* pt;
        float* eta;
        int* pdg_id;
        float* vx;
        float* vy;
        float* vz;

        // Space points
        int nSP;
        int* SPindex;
        double* SPx;
        double* SPy;
        double* SPz;
        int* SPCL1_index;
        int* SPCL2_index;
        int* SPisOverlap;

        // Clusters
        int nCL;
        int* CLindex;
        std::vector<std::string>* CLhardware;
        int* CLbarrel_endcap;
        int* CLeta_module;
        int* CLphi_module;
        ULong64_t* CLmoduleID;
        std::vector<std::vector<int>>* CLparticleLink_eventIndex;
        std::vector<std::vector<int>>* CLparticleLink_barcode;

        TTree_dump(TTree* t);
        ~TTree_dump();

        void GetEntry(Long64_t entry);
    
    private:
        void Init();
};
