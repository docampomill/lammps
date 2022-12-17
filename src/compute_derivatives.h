/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef COMPUTE_CLASS

ComputeStyle(derivatives,ComputeDerivatives)

#else

#ifndef COMPUTE_DERIVATIVES_H
#define COMPUTE_DERIVATIVES_H

#include "compute.h"

namespace LAMMPS_NS {

class ComputeDerivatives : public Compute {
 public: 
  ComputeDerivatives(class LAMMPS *, int, char **);
  ~ComputeDerivatives();
  void init();
  void init_list(int, class NeighList *);
  void compute_local();
  double memory_usage();

 private:
  double cutsq;
  double *eta_G2;
  double *zeta;
  double *eta_G4;
  int *lambda;
  int n_etaG2;
  int n_etaG4;
  int n_zeta;
  int n_lambda;
  int g2_flag;
  int g4_flag;

  class NeighList *list;
  double **alocal;
  int nmax_local;

  void compute_derivatives();
  void reallocate(int);
};

}

#endif
#endif

// compute ID group-ID derivatives Rc keyword values ... end
// Keyword options: etaG2, etaG4, zeta, lambda

