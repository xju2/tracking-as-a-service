/***********************************************************************
 * Module Map Graph library
 * Track reconstruction for HEP
 * copyright © 2022,2023 ROUGIER Charline
 * copyright © 2022,2023,2024 COLLARD Christophe
 * Laboratoire des 2 Infinis - Toulouse - France (L2IT)
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 *
 * contributors:
 * 2024 Alexis Vallier
 **********************************************************************/

#include "MMG/module_doublet.hpp"

//----------------------------------
template <class T>
module_doublet<T>::module_doublet ()
//----------------------------------
{
  _cuts_min = geometric_cuts<T> (-999, -999, -999, -999);
  _cuts_max = geometric_cuts<T> (999, 999, 999, 999);
  _cuts_sum = geometric_cuts<T> (0, 0, 0, 0);
  _cuts_sumSq = geometric_cuts<T> (0, 0, 0, 0);
}


//---------------------------------------------------------------
template <class T>
module_doublet<T>::module_doublet (const geometric_cuts<T>& cuts)
//---------------------------------------------------------------
{
  _cuts_min = _cuts_max = cuts;
  _cuts_sum = cuts;
  _cuts_sumSq = geometric_cuts<T> (0, 0, 0, 0);
  _cuts_sumSq.sumSq (cuts);
}


//-----------------------------------------------------------------------------
template <class Tf>
bool operator == (const module_doublet<Tf>& mt1, const module_doublet<Tf>& mt2)
//-----------------------------------------------------------------------------
{
  return (mt1._cuts_min == mt2._cuts_min) && (mt1._cuts_max == mt2._cuts_max);
}


//-----------------------------------------------------------------------------
template <class Tf>
bool operator != (const module_doublet<Tf>& md1, const module_doublet<Tf>& md2)
//-----------------------------------------------------------------------------
{
  return !(md1 == md2);
}


//-----------------------------------------------------------------------
template <class Tf>
std::ostream& operator << (std::ostream& s, const module_doublet<Tf>& md)
//-----------------------------------------------------------------------
{
//  s << "occurence # " << mt.occurence << std::endl;
  s << "z0 max 12 = " << md._cuts_max.z0() << std::endl;
  s << "z0 min 12 = " << md._cuts_min.z0() << std::endl;
  s << "d_phi max 12 = " << md._cuts_max.d_phi() << std::endl;
  s << "d_phi min 12 = " << md._cuts_min.d_phi() << std::endl;
  s << "phi slope max 12 = " << md._cuts_max.phi_slope() << std::endl;
  s << "phi slope min 12 = " << md._cuts_min.phi_slope() << std::endl;
  s << "delta max 12 = " << md._cuts_max.d_eta() << std::endl;
  s << "delta min 12 = " << md._cuts_min.d_eta() << std::endl;

  return s;
}


//------------------------------------------------------------------------------------
template <class T>
void module_doublet<T>::branch (TTree* treeModuleDoublets, const std::string& modules)
//------------------------------------------------------------------------------------
{
  _cuts_min.branch (treeModuleDoublets, modules, "min");
  _cuts_max.branch (treeModuleDoublets, modules, "max");
  _cuts_sum.branch (treeModuleDoublets, modules, "sum");
  _cuts_sumSq.branch (treeModuleDoublets, modules, "sumSq");
}


//------------------------------------------------------------------------------------------------
template <class T>
void module_doublet<T>::set_branch_address (TTree* treeModuleDoublets, const std::string& modules)
//------------------------------------------------------------------------------------------------
{
  _cuts_min.set_branch_address (treeModuleDoublets, modules, "min");
  _cuts_max.set_branch_address (treeModuleDoublets, modules, "max");

  // For backward compatibility, if sum and sumSq branches are not present
  // do not try to read them
  TObjArray* branches = treeModuleDoublets->GetListOfBranches();
  if (branches->FindObject ( (std::string ("z0sum") + std::string("_") + modules).c_str() )) {
    _cuts_sum.set_branch_address (treeModuleDoublets, modules, "sum");
    _cuts_sumSq.set_branch_address (treeModuleDoublets, modules, "sumSq");
  }
}


//-------------------------------------------------------------------
template <class T>
bool module_doublet<T>::add_occurence (const geometric_cuts<T>& cuts)
//-------------------------------------------------------------------
{
  if (!cuts.check_cut_values()) return false;
  _cuts_min.min (cuts);
  _cuts_max.max (cuts);
  _cuts_sum.sum (cuts);
  _cuts_sumSq.sumSq (cuts);

  return true;
}


//-----------------------------------------------------------------------------
template <class T>
void module_doublet<T>::add_occurences (const module_doublet<T>& ModuleDoublet)
//-----------------------------------------------------------------------------
{
  _cuts_min.min (ModuleDoublet._cuts_min);
  _cuts_max.max (ModuleDoublet._cuts_max);
  _cuts_sum.sum (ModuleDoublet._cuts_sum);
  _cuts_sumSq.sum (ModuleDoublet._cuts_sumSq);

  // If the ModuleDoublet stroe already a sum of squares (and not only one idivial square)
  // we cannot do sumSq1 += sumSq2*sumSq2
  // we must do sumSq1 += sumSq2 (ie use sum() and not sumSq())
}


//--------------------------------------------
template <class T>
void module_doublet<T>::add_numerical_error ()
//--------------------------------------------
{
  _cuts_min.add_numerical_error (-epsilon<T>());
  _cuts_max.add_numerical_error (epsilon<T>());
}
