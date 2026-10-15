/***********************************************************************
 * Module Map Graph library
 * Track reconstruction for HEP
 * copyright © 2023 TORRES Heberth
 * copyright © 2026 COLLARD Christophe
 * Laboratoire des 2 Infinis - Toulouse - France (L2IT)
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 **********************************************************************/

#include <MMG/strip_module_DB.hpp>


//---------------------------------------------------------------------------------------------------------------------
template <class T>
strip_module_DB<T>::strip_module_DB (const std::string& dbFilePath){
//---------------------------------------------------------------------------------------------------------------------
  TChain tree ("stripModules");
  TString TdbFilePath (dbFilePath);
  tree.Add (TdbFilePath);
  tree.SetEstimate (tree.GetEntries()+1);

  _nMods = tree.Draw ("x0:y0:z0:barrel_endcap:layer_disk:eta_module:phi_module", "", "goff");

  for (int i=0; i<_nMods; i++) {
    db_x0           .push_back(tree.GetVal(0)[i]);
    db_y0           .push_back(tree.GetVal(1)[i]);
    db_z0           .push_back(tree.GetVal(2)[i]);
    db_barrel_endcap.push_back(tree.GetVal(3)[i]);
    db_layer_disk   .push_back(tree.GetVal(4)[i]);
    db_eta_module   .push_back(tree.GetVal(5)[i]);
    db_phi_module   .push_back(tree.GetVal(6)[i]);
  }
}

//---------------------------------------------------------------------------------------------------------------------
template <class T>
std::vector<T> strip_module_DB<T>::get_module_position (const int& barrel_endcap, const int& layer_disk,
                                                        const int& eta_module, const int& phi_module){
//---------------------------------------------------------------------------------------------------------------------
  if (barrel_endcap)
    return std::vector<T> (3, -9999.);

  for (int i=0; i<_nMods; i++)
    if (db_barrel_endcap[i] == barrel_endcap  &&  db_layer_disk[i] == layer_disk &&
        db_eta_module[i] == eta_module  &&  db_phi_module[i] == phi_module)
      return std::vector<T> {db_x0[i], db_y0[i], db_z0[i]};

  std::cout << "ERROR: StripModuleDB::GetModPosition: Module not found: "
            << "\tSelection was: barrel_endcap = " << barrel_endcap
            << " && layer_disk = " << layer_disk
            << " && eta_module = " << eta_module
            << " && phi_module = " << phi_module
            << std::endl;
  exit(1);
}

//---------------------------------------------------------------------------------------------------------------------
template <class T>
std::vector<T> strip_module_DB<T>::get_module_position (const std::string & hardware, const int& barrel_endcap,
                                                  const int& layer_disk, const int& eta_module, const int& phi_module){
//---------------------------------------------------------------------------------------------------------------------
  std::vector<T> ret;
  ret.assign (3, -9999);

  if (hardware == "STRIP") ret = get_module_position (barrel_endcap, layer_disk, eta_module, phi_module);

  return ret;
}

//---------------------------------------------------------------------------------------------------------------------
template <class T>
template <class DT>
std::vector<T> strip_module_DB<T>::get_module_position (const int& it, const TTree_hits<DT>& TThits){
//---------------------------------------------------------------------------------------------------------------------
  TThits.get (it);
  if (TThits.hardware() == "STRIP")
    return get_module_position (TThits.barrel_endcap(), TThits.layer_disk(), TThits.eta_module(), TThits.phi_module());
  else return std::vector<T> (3, -9999);
}
