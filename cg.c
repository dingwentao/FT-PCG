
/*
    This file implements the conjugate gradient method in PETSc as part of
    KSP. You can use this as a starting point for implementing your own
    Krylov method that is not provided with PETSc.

    The following basic routines are required for each Krylov method.
        KSPCreate_XXX()          - Creates the Krylov context
        KSPSetFromOptions_XXX()  - Sets runtime options
        KSPSolve_XXX()           - Runs the Krylov method
        KSPDestroy_XXX()         - Destroys the Krylov context, freeing all
                                   memory it needed
    Here the "_XXX" denotes a particular implementation, in this case
    we use _CG (e.g. KSPCreate_CG, KSPDestroy_CG). These routines are
    are actually called vai the common user interface routines
    KSPSetType(), KSPSetFromOptions(), KSPSolve(), and KSPDestroy() so the
    application code interface remains identical for all preconditioners.

    Other basic routines for the KSP objects include
        KSPSetUp_XXX()
        KSPView_XXX()             - Prints details of solver being used.

    Detailed notes:
    By default, this code implements the CG (Conjugate Gradient) method,
    which is valid for real symmetric (and complex Hermitian) positive
    definite matrices. Note that for the complex Hermitian case, the
    VecDot() arguments within the code MUST remain in the order given
    for correct computation of inner products.

    Reference: Hestenes and Steifel, 1952.

    By switching to the indefinite vector inner product, VecTDot(), the
    same code is used for the complex symmetric case as well.  The user
    must call KSPCGSetType(ksp,KSP_CG_SYMMETRIC) or use the option
    -ksp_cg_type symmetric to invoke this variant for the complex case.
    Note, however, that the complex symmetric code is NOT valid for
    all such matrices ... and thus we don't recommend using this method.
*/
/*
       cgimpl.h defines the simple data structured used to store information
    related to the type of matrix (e.g. complex symmetric) being solved and
    data used during the optional Lanczo process used to compute eigenvalues
*/
#include <time.h>
#include <../src/ksp/ksp/impls/cg/cgimpl.h>       /*I "petscksp.h" I*/
extern PetscErrorCode KSPComputeExtremeSingularValues_CG(KSP,PetscReal*,PetscReal*);
extern PetscErrorCode KSPComputeEigenvalues_CG(KSP,PetscInt,PetscReal*,PetscReal*,PetscInt*);

/*
     KSPSetUp_CG - Sets up the workspace needed by the CG method.

      This is called once, usually automatically by KSPSolve() or KSPSetUp()
     but can be called directly by KSPSetUp()
*/
#undef __FUNCT__
#define __FUNCT__ "KSPSetUp_CG"
PetscErrorCode KSPSetUp_CG(KSP ksp)
{
  KSP_CG         *cgP = (KSP_CG*)ksp->data;
  PetscErrorCode ierr;
  /* Dingwen */
  PetscInt			 maxit = ksp->max_it,nwork = 11; /* add predefined vectors, checksum(A) and checkpoint vectors */
  //  PetscInt       maxit = ksp->max_it,nwork = 3;

  PetscFunctionBegin;
  /* get work vectors needed by CG */
  if (cgP->singlereduction) nwork += 2;
  ierr = KSPSetWorkVecs(ksp,nwork);CHKERRQ(ierr);

  /*
     If user requested computations of eigenvalues then allocate work
     work space needed
  */
  if (ksp->calc_sings) {
    /* get space to store tridiagonal matrix for Lanczos */
    ierr = PetscMalloc4(maxit+1,&cgP->e,maxit+1,&cgP->d,maxit+1,&cgP->ee,maxit+1,&cgP->dd);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory((PetscObject)ksp,2*(maxit+1)*(sizeof(PetscScalar)+sizeof(PetscReal)));CHKERRQ(ierr);

    ksp->ops->computeextremesingularvalues = KSPComputeExtremeSingularValues_CG;
    ksp->ops->computeeigenvalues           = KSPComputeEigenvalues_CG;
  }
  PetscFunctionReturn(0);
}

/*
       KSPSolve_CG - This routine actually applies the conjugate gradient  method

   This routine is MUCH too messy. I has too many options (norm type and single reduction) embedded making the code confusing and likely to be buggy.

   Input Parameter:
.     ksp - the Krylov space object that was set to use conjugate gradient, by, for
            example, KSPCreate(MPI_Comm,KSP *ksp); KSPSetType(ksp,KSPCG);
*/
#undef __FUNCT__
#define __FUNCT__ "KSPSolve_CG"
PetscErrorCode  KSPSolve_CG(KSP ksp)
{
  PetscErrorCode ierr;
  PetscInt       i,stored_max_it,eigs;
  PetscScalar    dpi = 0.0,a = 1.0,beta,betaold = 1.0,b = 0,*e = 0,*d = 0,delta,dpiold;
  PetscReal      dp  = 0.0;
  Vec            X,B,Z,R,P,S,W;
  KSP_CG         *cg;
  Mat            Amat,Pmat;
  PetscBool      diagonalscale;
  /* Dingwen */
  PetscScalar	CKSX,CKSZ,CKSR,CKSP,CKSW;
  PetscScalar	CKSX1,CKSZ1,CKSR1,CKSP1,CKSS1,CKSW1;
  PetscScalar	CKSX2,CKSZ2,CKSR2,CKSP2,CKSS2,CKSW2;
  PetscScalar	CKSX3,CKSZ3,CKSR3,CKSP3,CKSS3,CKSW3;
  Vec			CKSAmat1;
  Vec			CKSAmat2;
  Vec			CKSAmat3;
  Vec			C1,C2,C3;
  PetscScalar	d1,d2,d3;
  Vec			CKSAmat;
  PetscScalar	sumX,sumZ,sumR,sumP,sumW;
  Vec			CKPX,CKPP;
  PetscScalar	CKPbetaold;
  PetscInt		CKPi;
  PetscBool		flag1 = PETSC_TRUE,flag2 = PETSC_TRUE,flag3 = PETSC_TRUE,flag4 = PETSC_TRUE,flag5 = PETSC_TRUE,
				flag6 = PETSC_TRUE,flag7 = PETSC_TRUE,flag8 = PETSC_TRUE,flag9 = PETSC_TRUE,flag10 = PETSC_TRUE;
  PetscScalar	v;
  PetscInt		pos1,pos2;
  PetscInt		solver_type, error_type;
  PetscInt		itv_c, itv_d;
  PetscInt		inj_itr, inj_times;
  PetscInt		rank,size;
  /* Dingwen */
  
  PetscFunctionBegin;
  #define VecXDot(x,y,a) (((cg->type) == (KSP_CG_HERMITIAN)) ? VecDot(x,y,a) : VecTDot(x,y,a))
  /* Dingwen */
  MPI_Comm_rank	(MPI_COMM_WORLD,&rank);
  MPI_Comm_size	(MPI_COMM_WORLD,&size);
  solver_type	= ksp->solver_type;
  error_type 	= ksp->error_type;
  itv_c 		= ksp->itv_c;
  itv_d 		= ksp->itv_d;
  inj_itr 		= ksp->inj_itr;
  inj_times 	= ksp->inj_times;
  /* Dingwen */

  
  ierr = PCGetDiagonalScale(ksp->pc,&diagonalscale);CHKERRQ(ierr);
  if (diagonalscale) SETERRQ1(PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"Krylov method %s does not support diagonal scaling",((PetscObject)ksp)->type_name);

  if(solver_type==0)
  {
	  cg            = (KSP_CG*)ksp->data;
  eigs          = ksp->calc_sings;
  stored_max_it = ksp->max_it;
  X             = ksp->vec_sol;
  B             = ksp->vec_rhs;
  R             = ksp->work[0];
  Z             = ksp->work[1];
  P             = ksp->work[2];
  if (cg->singlereduction) {
    S = ksp->work[3];
    W = ksp->work[4];
  } else {
    S = 0;                      /* unused */
    W = Z;
  }

  if (eigs) {e = cg->e; d = cg->d; e[0] = 0.0; }
  ierr = PCGetOperators(ksp->pc,&Amat,&Pmat);CHKERRQ(ierr);

  ksp->its = 0;
  if (!ksp->guess_zero) {
    ierr = KSP_MatMult(ksp,Amat,X,R);CHKERRQ(ierr);            /*     r <- b - Ax     */
    ierr = VecAYPX(R,-1.0,B);CHKERRQ(ierr);
  } else {
    ierr = VecCopy(B,R);CHKERRQ(ierr);                         /*     r <- b (x is 0) */
  }

  switch (ksp->normtype) {
  case KSP_NORM_PRECONDITIONED:
    ierr = KSP_PCApply(ksp,R,Z);CHKERRQ(ierr);                   /*     z <- Br         */
    ierr = VecNorm(Z,NORM_2,&dp);CHKERRQ(ierr);                /*    dp <- z'*z = e'*A'*B'*B*A'*e'     */
    break;
  case KSP_NORM_UNPRECONDITIONED:
    ierr = VecNorm(R,NORM_2,&dp);CHKERRQ(ierr);                /*    dp <- r'*r = e'*A'*A*e            */
    break;
  case KSP_NORM_NATURAL:
    ierr = KSP_PCApply(ksp,R,Z);CHKERRQ(ierr);                   /*     z <- Br         */
    if (cg->singlereduction) {
      ierr = KSP_MatMult(ksp,Amat,Z,S);CHKERRQ(ierr);
      ierr = VecXDot(Z,S,&delta);CHKERRQ(ierr);
    }
    ierr = VecXDot(Z,R,&beta);CHKERRQ(ierr);                     /*  beta <- z'*r       */
    KSPCheckDot(ksp,beta);
    dp = PetscSqrtReal(PetscAbsScalar(beta));                           /*    dp <- r'*z = r'*B*r = e'*A'*B*A*e */
    break;
  case KSP_NORM_NONE:
    dp = 0.0;
    break;
  default: SETERRQ1(PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"%s",KSPNormTypes[ksp->normtype]);
  }
  ierr       = KSPLogResidualHistory(ksp,dp);CHKERRQ(ierr);
  ierr       = KSPMonitor(ksp,0,dp);CHKERRQ(ierr);
  ksp->rnorm = dp;

  ierr = (*ksp->converged)(ksp,0,dp,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);      /* test for convergence */
  if (ksp->reason) PetscFunctionReturn(0);

  if (ksp->normtype != KSP_NORM_PRECONDITIONED && (ksp->normtype != KSP_NORM_NATURAL)) {
    ierr = KSP_PCApply(ksp,R,Z);CHKERRQ(ierr);                   /*     z <- Br         */
  }
  if (ksp->normtype != KSP_NORM_NATURAL) {
    if (cg->singlereduction) {
      ierr = KSP_MatMult(ksp,Amat,Z,S);CHKERRQ(ierr);
      ierr = VecXDot(Z,S,&delta);CHKERRQ(ierr);
    }
    ierr = VecXDot(Z,R,&beta);CHKERRQ(ierr);         /*  beta <- z'*r       */
    KSPCheckDot(ksp,beta);
  }

  struct timespec start, end;
  long long int local_diff, global_diff;
  clock_gettime(CLOCK_REALTIME, &start);

  i = 0;
  do {
    ksp->its = i+1;
    if (beta == 0.0) {
      ksp->reason = KSP_CONVERGED_ATOL;
      ierr        = PetscInfo(ksp,"converged due to beta = 0\n");CHKERRQ(ierr);
      break;
#if !defined(PETSC_USE_COMPLEX)
    } else if ((i > 0) && (beta*betaold < 0.0)) {
      ksp->reason = KSP_DIVERGED_INDEFINITE_PC;
      ierr        = PetscInfo(ksp,"diverging due to indefinite preconditioner\n");CHKERRQ(ierr);
      break;
#endif
    }
    if (!i) {
      ierr = VecCopy(Z,P);CHKERRQ(ierr);         /*     p <- z          */
      b    = 0.0;
    } else {
      b = beta/betaold;
      if (eigs) {
        if (ksp->max_it != stored_max_it) SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"Can not change maxit AND calculate eigenvalues");
        e[i] = PetscSqrtReal(PetscAbsScalar(b))/a;
      }
      ierr = VecAYPX(P,b,Z);CHKERRQ(ierr);    /*     p <- z + b* p   */
    }
    dpiold = dpi;
    if (!cg->singlereduction || !i) {
      ierr = KSP_MatMult(ksp,Amat,P,W);CHKERRQ(ierr);          /*     w <- Ap         */
      ierr = VecXDot(P,W,&dpi);CHKERRQ(ierr);                  /*     dpi <- p'w     */
    } else {
      ierr = VecAYPX(W,beta/betaold,S);CHKERRQ(ierr);                  /*     w <- Ap         */
      dpi  = delta - beta*beta*dpiold/(betaold*betaold);             /*     dpi <- p'w     */
    }
    betaold = beta;
    KSPCheckDot(ksp,beta);

    if ((dpi == 0.0) || ((i > 0) && (PetscRealPart(dpi*dpiold) <= 0.0))) {
      ksp->reason = KSP_DIVERGED_INDEFINITE_MAT;
      ierr        = PetscInfo(ksp,"diverging due to indefinite or negative definite matrix\n");CHKERRQ(ierr);
      break;
    }
    a = beta/dpi;                                 /*     a = beta/p'w   */
    if (eigs) d[i] = PetscSqrtReal(PetscAbsScalar(b))*e[i] + 1.0/a;
    ierr = VecAXPY(X,a,P);CHKERRQ(ierr);          /*     x <- x + ap     */
    ierr = VecAXPY(R,-a,W);CHKERRQ(ierr);                      /*     r <- r - aw    */
    if (ksp->normtype == KSP_NORM_PRECONDITIONED && ksp->chknorm < i+2) {
      ierr = KSP_PCApply(ksp,R,Z);CHKERRQ(ierr);               /*     z <- Br         */
      if (cg->singlereduction) {
        ierr = KSP_MatMult(ksp,Amat,Z,S);CHKERRQ(ierr);
      }
      ierr = VecNorm(Z,NORM_2,&dp);CHKERRQ(ierr);              /*    dp <- z'*z       */
    } else if (ksp->normtype == KSP_NORM_UNPRECONDITIONED && ksp->chknorm < i+2) {
      ierr = VecNorm(R,NORM_2,&dp);CHKERRQ(ierr);              /*    dp <- r'*r       */
    } else if (ksp->normtype == KSP_NORM_NATURAL) {
      ierr = KSP_PCApply(ksp,R,Z);CHKERRQ(ierr);               /*     z <- Br         */
      if (cg->singlereduction) {
        PetscScalar tmp[2];
        Vec         vecs[2];
        vecs[0] = S; vecs[1] = R;
        ierr    = KSP_MatMult(ksp,Amat,Z,S);CHKERRQ(ierr);
        ierr  = VecMDot(Z,2,vecs,tmp);CHKERRQ(ierr);
        delta = tmp[0]; beta = tmp[1];
      } else {
        ierr = VecXDot(Z,R,&beta);CHKERRQ(ierr);     /*  beta <- r'*z       */
      }
      KSPCheckDot(ksp,beta);
      dp = PetscSqrtReal(PetscAbsScalar(beta));
    } else {
      dp = 0.0;
    }
    ksp->rnorm = dp;
    CHKERRQ(ierr);KSPLogResidualHistory(ksp,dp);CHKERRQ(ierr);
    ierr = KSPMonitor(ksp,i+1,dp);CHKERRQ(ierr);
    ierr = (*ksp->converged)(ksp,i+1,dp,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
    if (ksp->reason) break;

    if ((ksp->normtype != KSP_NORM_PRECONDITIONED && (ksp->normtype != KSP_NORM_NATURAL)) || (ksp->chknorm >= i+2)) {
      ierr = KSP_PCApply(ksp,R,Z);CHKERRQ(ierr);                   /*     z <- Br         */
      if (cg->singlereduction) {
        ierr = KSP_MatMult(ksp,Amat,Z,S);CHKERRQ(ierr);
      }
    }
    if ((ksp->normtype != KSP_NORM_NATURAL) || (ksp->chknorm >= i+2)) {
      if (cg->singlereduction) {
        PetscScalar tmp[2];
        Vec         vecs[2];
        vecs[0] = S; vecs[1] = R;
        ierr  = VecMDot(Z,2,vecs,tmp);CHKERRQ(ierr);
        delta = tmp[0]; beta = tmp[1];
      } else {
        ierr = VecXDot(Z,R,&beta);CHKERRQ(ierr);        /*  beta <- z'*r       */
      }
      KSPCheckDot(ksp,beta);
    }

    i++;
  } while (i<ksp->max_it);
  clock_gettime(CLOCK_REALTIME, &end);
  local_diff = 1000000000L*(end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec);
  MPI_Reduce(&local_diff, &global_diff, 1, MPI_LONG_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  PetscPrintf(MPI_COMM_WORLD,"elapsed time of main loop = %lf nanoseconds\n", (double) (global_diff)/size);
  }
  
  if (solver_type==1)
  {
	  cg            = (KSP_CG*)ksp->data;
  eigs          = ksp->calc_sings;
  stored_max_it = ksp->max_it;
  X             = ksp->vec_sol;
  B             = ksp->vec_rhs;
  R             = ksp->work[0];
  Z             = ksp->work[1];
  P             = ksp->work[2];
  /* Dingwen */
  CKPX			= ksp->work[3];
  CKPP			= ksp->work[4];
  CKSAmat		= ksp->work[5];
  C1			= ksp->work[6];
  /* Dingwen */
  
  if (cg->singlereduction) {
    S = ksp->work[8];
    W = ksp->work[9];
  } else {
    S = 0;                      /* unused */
    W = Z;
  }
    
  if (eigs) {e = cg->e; d = cg->d; e[0] = 0.0; }
  ierr = PCGetOperators(ksp->pc,&Amat,&Pmat);CHKERRQ(ierr);
  
  ksp->its = 0;
  if (!ksp->guess_zero) {
    ierr = KSP_MatMult(ksp,Amat,X,R);CHKERRQ(ierr);            /*     r <- b - Ax     */
    ierr = VecAYPX(R,-1.0,B);CHKERRQ(ierr);
  } else {
    ierr = VecCopy(B,R);CHKERRQ(ierr);                         /*     r <- b (x is 0) */
  }
  

  /* Dingwen */	
  /* checksum coefficients initialization */
  PetscInt n;
  PetscInt *index;
  PetscScalar *v1;
  ierr = VecGetSize(B,&n);
  v1 	= (PetscScalar *)malloc(n*sizeof(PetscScalar));
  index	= (PetscInt *)malloc(n*sizeof(PetscInt));
  for (i=0; i<n; i++)
  {
	  index[i] = i;
	  v1[i] = 1.0;
  }
  ierr	= VecSetValues(C1,n,index,v1,INSERT_VALUES);CHKERRQ(ierr);	
  d1 = 1.0;
  /* Dingwen */

  switch (ksp->normtype) {
  case KSP_NORM_PRECONDITIONED:
    ierr = KSP_PCApply(ksp,R,Z);CHKERRQ(ierr);                   /*     z <- Br         */
    ierr = VecNorm(Z,NORM_2,&dp);CHKERRQ(ierr);                /*    dp <- z'*z = e'*A'*B'*B*A'*e'     */
    break;
  case KSP_NORM_UNPRECONDITIONED:
    ierr = VecNorm(R,NORM_2,&dp);CHKERRQ(ierr);                /*    dp <- r'*r = e'*A'*A*e            */
    break;
  case KSP_NORM_NATURAL:
    ierr = KSP_PCApply(ksp,R,Z);CHKERRQ(ierr);                   /*     z <- Br         */
    if (cg->singlereduction) {
      ierr = KSP_MatMult(ksp,Amat,Z,S);CHKERRQ(ierr);
      ierr = VecXDot(Z,S,&delta);CHKERRQ(ierr);
	}
    ierr = VecXDot(Z,R,&beta);CHKERRQ(ierr);                     /*  beta <- z'*r       */
    KSPCheckDot(ksp,beta);
    dp = PetscSqrtReal(PetscAbsScalar(beta));                           /*    dp <- r'*z = r'*B*r = e'*A'*B*A*e */
    break;
  case KSP_NORM_NONE:
    dp = 0.0;
    break;
  default: SETERRQ1(PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"%s",KSPNormTypes[ksp->normtype]);
  }
  
  ierr       = KSPLogResidualHistory(ksp,dp);CHKERRQ(ierr);
  ierr       = KSPMonitor(ksp,0,dp);CHKERRQ(ierr);
  ksp->rnorm = dp;

  ierr = (*ksp->converged)(ksp,0,dp,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);      /* test for convergence */
  if (ksp->reason) PetscFunctionReturn(0);

  if (ksp->normtype != KSP_NORM_PRECONDITIONED && (ksp->normtype != KSP_NORM_NATURAL)) {
    ierr = KSP_PCApply(ksp,R,Z);CHKERRQ(ierr);                   /*     z <- Br         */
  }
  if (ksp->normtype != KSP_NORM_NATURAL) {
    if (cg->singlereduction) {
      ierr = KSP_MatMult(ksp,Amat,Z,S);CHKERRQ(ierr);
      ierr = VecXDot(Z,S,&delta);CHKERRQ(ierr);
    }
    ierr = VecXDot(Z,R,&beta);CHKERRQ(ierr);         /*  beta <- z'*r       */
    KSPCheckDot(ksp,beta);
  }

  /* Dingwen */
  /* Checksum Initialization */
  ierr = VecXDot(C1,X,&CKSX);CHKERRQ(ierr);						/* Compute the initial checksum(X) */ 
  ierr = VecXDot(C1,W,&CKSW);CHKERRQ(ierr);						/* Compute the initial checksum(W) */
  ierr = VecXDot(C1,R,&CKSR);CHKERRQ(ierr);						/* Compute the initial checksum(R) */
  ierr = VecXDot(C1,Z,&CKSZ);CHKERRQ(ierr);						/* Compute the initial checksum(Z) */
  ierr = KSP_MatMultTranspose(ksp,Amat,C1,CKSAmat);CHKERRQ(ierr);
  ierr = VecAXPY(CKSAmat,-d1,C1);CHKERRQ(ierr);					/* Compute the initial checksum(A) */
  /* Dingwen */
  

  struct timespec start, end;
  long long int local_diff, global_diff;
  
  struct timespec start_d, end_d;
  struct timespec start_r, end_r;
  struct timespec start_c, end_c;
  struct timespec start_i, end_i;
  long long int time_d = 0,time_r = 0,time_c = 0,time_i = 0;

  clock_gettime(CLOCK_REALTIME, &start);
  i = 0;
  PetscInt	numd = 0, numr = 0, numc = 0, numi = 0;
  do {
	  /* Dingwen */
	  if ((i>0) && (i%itv_d == 0))
	  {
		  clock_gettime(CLOCK_REALTIME, &start_d);
		  ierr = VecXDot(C1,X,&sumX);CHKERRQ(ierr);
		  ierr = VecXDot(C1,R,&sumR);CHKERRQ(ierr);
		  clock_gettime(CLOCK_REALTIME, &end_d);
		  time_d += 1000000000L*(end_d.tv_sec - start_d.tv_sec) + (end_d.tv_nsec - start_d.tv_nsec);
		  numd++;
		  if ((PetscAbsScalar(sumX-CKSX)/(n*n) > 1.0e-10) || (PetscAbsScalar(sumR-CKSR)/(n*n) > 1.0e-10))
		  {
			  clock_gettime(CLOCK_REALTIME, &start_r);
			  /* Rollback and Recovery */
			  PetscPrintf(MPI_COMM_WORLD,"Recovery start...\n");
			  PetscPrintf(MPI_COMM_WORLD,"Rollback from iteration-%d to iteration-%d\n",i,CKPi);
			  betaold = CKPbetaold;										/* Recovery scalar betaold by checkpoint*/
			  i = CKPi;													/* Recovery integer i by checkpoint */
			  ierr = VecCopy(CKPP,P);CHKERRQ(ierr);						/* Recovery vector P from checkpoint */
			  ierr = VecXDot(C1,P,&CKSP);CHKERRQ(ierr);					/* Recovery checksum(P) by P */ 
			  ierr = KSP_MatMult(ksp,Amat,P,W);CHKERRQ(ierr);			/* Recovery vector W by P */
			  ierr = VecXDot(P,W,&dpi);CHKERRQ(ierr);					/* Recovery scalar dpi by P and W */
			  ierr = VecCopy(CKPX,X);CHKERRQ(ierr);						/* Recovery vector X from checkpoint */
			  ierr = VecXDot(C1,X,&CKSX);CHKERRQ(ierr);					/* Recovery checksum(X) by X */ 
			  ierr = KSP_MatMult(ksp,Amat,X,R);CHKERRQ(ierr);			/* Recovery vector R by X */
			  ierr = VecAYPX(R,-1.0,B);CHKERRQ(ierr);
			  ierr = VecXDot(C1,R,&CKSR);CHKERRQ(ierr);					/* Recovery checksum(R) by R */
			  ierr = KSP_PCApply(ksp,R,Z);CHKERRQ(ierr);				/* Recovery vector Z by R */
			  ierr = VecXDot(C1,Z,&CKSZ);CHKERRQ(ierr);					/* Recovery checksum(Z) by Z */
			  ierr = VecXDot(Z,R,&beta);CHKERRQ(ierr);					/* Recovery scalar beta by Z and R */
			  PetscPrintf(MPI_COMM_WORLD,"Recovery end.\n");
			  clock_gettime(CLOCK_REALTIME, &end_r);
			  time_r += 1000000000L*(end_r.tv_sec - start_r.tv_sec) + (end_r.tv_nsec - start_r.tv_nsec);
			  numr++;
		}
		else if (i%(itv_c*itv_d) == 0)
		{
			clock_gettime(CLOCK_REALTIME, &start_c);
			ierr = VecCopy(X,CKPX);CHKERRQ(ierr);
			ierr = VecCopy(P,CKPP);CHKERRQ(ierr);
			CKPbetaold = betaold;
			CKPi = i;
			clock_gettime(CLOCK_REALTIME, &end_c);
			time_c += 1000000000L*(end_c.tv_sec - start_c.tv_sec) + (end_c.tv_nsec - start_c.tv_nsec);
			numc++;		
		}
	}
	
	clock_gettime(CLOCK_REALTIME, &start_i);

	  ksp->its++;
	  if (beta == 0.0) {
      ksp->reason = KSP_CONVERGED_ATOL;
      ierr        = PetscInfo(ksp,"converged due to beta = 0\n");CHKERRQ(ierr);
      break;
#if !defined(PETSC_USE_COMPLEX)
    } else if ((i > 0) && (beta*betaold < 0.0)) {
      ksp->reason = KSP_DIVERGED_INDEFINITE_PC;
      ierr        = PetscInfo(ksp,"diverging due to indefinite preconditioner\n");CHKERRQ(ierr);
      break;
#endif
    }
    if (!i) {
      ierr = VecCopy(Z,P);CHKERRQ(ierr);         /*     p <- z          */
      b    = 0.0;
	  /* Dingwen */
	  ierr = VecXDot(C1,P, &CKSP);CHKERRQ(ierr);  				/* Compute the initial checksum(P) */
	  /* Dingwen */
    } else {
      b = beta/betaold;
      if (eigs) {
        if (ksp->max_it != stored_max_it) SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"Can not change maxit AND calculate eigenvalues");
        e[i] = PetscSqrtReal(PetscAbsScalar(b))/a;
      }
      ierr = VecAYPX(P,b,Z);CHKERRQ(ierr);    /*     p <- z + b* p   */	  
	  /* Dingwen */
	  CKSP = CKSZ + b*CKSP;										/* Update checksum(P) = checksum(Z) + b*checksum(P); */
	  /* Dingwen */
    }
    dpiold = dpi;
    if (!cg->singlereduction || !i) {
      ierr = KSP_MatMult(ksp,Amat,P,W);CHKERRQ(ierr);          /*     w <- Ap         */	/* MVM */
      ierr = VecXDot(P,W,&dpi);CHKERRQ(ierr);                  /*     dpi <- p'w     */	  
	  /* Dingwen */
	  ierr = VecXDot(CKSAmat, P, &CKSW);CHKERRQ(ierr);
	  CKSW = CKSW + d1*CKSP;									/* Update checksum(W) = checksum(A)P + d1*checksum(P); */
	  /* Dingwen */
	  
	  /* Inject an error in W */
	  if((i==inj_itr)&&(flag1)&&(error_type==1))
	  {
		  srand (time(NULL));
		  VecScatter	ctx;
		  Vec			W_SEQ;
		  PetscScalar	*W_ARR;
		  VecScatterCreateToAll(W,&ctx,&W_SEQ);
		  VecScatterBegin(ctx,W,W_SEQ,INSERT_VALUES,SCATTER_FORWARD);
		  VecScatterEnd(ctx,W,W_SEQ,INSERT_VALUES,SCATTER_FORWARD);
		  VecGetArray(W_SEQ,&W_ARR);
		  pos1 = rand()/(RAND_MAX/n+1);
		  v		= W_ARR[pos1]*inj_times;
		  ierr	= VecSetValues(W,1,&pos1,&v,INSERT_VALUES);CHKERRQ(ierr);
		  VecDestroy(&W_SEQ);
		  VecScatterDestroy(&ctx);
		  VecAssemblyBegin(W);
		  VecAssemblyEnd(W);
		  PetscPrintf(MPI_COMM_WORLD,"Inject an error in W of position-%d after MVM at iteration-%d\n", pos1,i);
		  flag1	= PETSC_FALSE;
	  }
	  
	  /* Inject two errors in W */
	  if((i==inj_itr)&&(flag2)&(error_type==2))
	  {
		  srand (time(NULL));
		  VecScatter	ctx;
		  Vec			W_SEQ;
		  PetscScalar	*W_ARR;
		  VecScatterCreateToAll(W,&ctx,&W_SEQ);
		  VecScatterBegin(ctx,W,W_SEQ,INSERT_VALUES,SCATTER_FORWARD);
		  VecScatterEnd(ctx,W,W_SEQ,INSERT_VALUES,SCATTER_FORWARD);
		  VecGetArray(W_SEQ,&W_ARR);
		  pos1 = rand()/(RAND_MAX/n+1);
		  v		= W_ARR[pos1]*inj_times;
		  ierr	= VecSetValues(W,1,&pos1,&v,INSERT_VALUES);CHKERRQ(ierr);
		  pos2 = rand()/(RAND_MAX/n+1);
		  v		= W_ARR[pos2]*inj_times;
		  ierr	= VecSetValues(W,1,&pos2,&v,INSERT_VALUES);CHKERRQ(ierr);
		  VecDestroy(&W_SEQ);
		  VecScatterDestroy(&ctx);
		  VecAssemblyBegin(W);
		  VecAssemblyEnd(W);
		  PetscPrintf(MPI_COMM_WORLD,"Inject two errors in W of position-%d and position-%d after MVM at iteration-%d\n", pos1,pos2,i);
		  flag2	= PETSC_FALSE;
		}
    } else {
      ierr = VecAYPX(W,beta/betaold,S);CHKERRQ(ierr);                  /*     w <- Ap         */
      dpi  = delta - beta*beta*dpiold/(betaold*betaold);             /*     dpi <- p'w     */
	}
    betaold = beta;
    KSPCheckDot(ksp,beta);

    if ((dpi == 0.0) || ((i > 0) && (PetscRealPart(dpi*dpiold) <= 0.0))) {
      ksp->reason = KSP_DIVERGED_INDEFINITE_MAT;
      ierr        = PetscInfo(ksp,"diverging due to indefinite or negative definite matrix\n");CHKERRQ(ierr);
      break;
    }
    a = beta/dpi;                                 /*     a = beta/p'w   */
    if (eigs) d[i] = PetscSqrtReal(PetscAbsScalar(b))*e[i] + 1.0/a;
    ierr = VecAXPY(X,a,P);CHKERRQ(ierr);          /*     x <- x + ap     */
	/* Dingwen */
	CKSX = CKSX + a*CKSP;									/* Update checksum(X) = checksum(X) + a*checksum(P); */
	/* Dingwen */
    
	ierr = VecAXPY(R,-a,W);CHKERRQ(ierr);                      /*     r <- r - aw    */

	/* Dingwen */
	CKSR = CKSR - a*CKSW;									/* Update checksum(R) = checksum(R) - a*checksum(W); */
	/* Dingwen */
	
	if (ksp->normtype == KSP_NORM_PRECONDITIONED && ksp->chknorm < i+2) {      
	  ierr = KSP_PCApply(ksp,R,Z);CHKERRQ(ierr);               /*     z <- Br         */
	  
	  /* Dingwen */
	  ierr = VecXDot(C1,Z, &CKSZ);CHKERRQ(ierr);				/* Update checksum(Z) */
	  /* Dingwen */
	  
	  if (cg->singlereduction) {
        ierr = KSP_MatMult(ksp,Amat,Z,S);CHKERRQ(ierr);			/* MVM */
      }
      ierr = VecNorm(Z,NORM_2,&dp);CHKERRQ(ierr);              /*    dp <- z'*z       */
    } else if (ksp->normtype == KSP_NORM_UNPRECONDITIONED && ksp->chknorm < i+2) {
      ierr = VecNorm(R,NORM_2,&dp);CHKERRQ(ierr);              /*    dp <- r'*r       */
    } else if (ksp->normtype == KSP_NORM_NATURAL) {
      ierr = KSP_PCApply(ksp,R,Z);CHKERRQ(ierr);               /*     z <- Br         */
 	  
	  /* Dingwen */
	  ierr = VecXDot(C1,Z, &CKSZ);CHKERRQ(ierr);				/* Update checksum(Z) */
	  /* Dingwen */
	  
	  if (cg->singlereduction) {
        PetscScalar tmp[2];
        Vec         vecs[2];
        vecs[0] = S; vecs[1] = R;
        ierr    = KSP_MatMult(ksp,Amat,Z,S);CHKERRQ(ierr);
        ierr  = VecMDot(Z,2,vecs,tmp);CHKERRQ(ierr);
        delta = tmp[0]; beta = tmp[1];
      } else {
        ierr = VecXDot(Z,R,&beta);CHKERRQ(ierr);     /*  beta <- r'*z       */
      }
      KSPCheckDot(ksp,beta);
      dp = PetscSqrtReal(PetscAbsScalar(beta));
    } else {
      dp = 0.0;
    }
	  
    ksp->rnorm = dp;
    CHKERRQ(ierr);KSPLogResidualHistory(ksp,dp);CHKERRQ(ierr);
    ierr = KSPMonitor(ksp,i+1,dp);CHKERRQ(ierr);
    ierr = (*ksp->converged)(ksp,i+1,dp,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
    if (ksp->reason) break;

    if ((ksp->normtype != KSP_NORM_PRECONDITIONED && (ksp->normtype != KSP_NORM_NATURAL)) || (ksp->chknorm >= i+2)) {
      ierr = KSP_PCApply(ksp,R,Z);CHKERRQ(ierr);                   /*     z <- Br         */
	  
	  /* Dingwen */
	  ierr = VecXDot(C1,Z, &CKSZ);CHKERRQ(ierr);				/* Update checksum(Z) */
	  /* Dingwen */
      
	  if (cg->singlereduction) {
        ierr = KSP_MatMult(ksp,Amat,Z,S);CHKERRQ(ierr);
      }
    }
		  
    if ((ksp->normtype != KSP_NORM_NATURAL) || (ksp->chknorm >= i+2)) {
      if (cg->singlereduction) {
        PetscScalar tmp[2];
        Vec         vecs[2];
        vecs[0] = S; vecs[1] = R;
        ierr  = VecMDot(Z,2,vecs,tmp);CHKERRQ(ierr);
        delta = tmp[0]; beta = tmp[1];
      } else {
        ierr = VecXDot(Z,R,&beta);CHKERRQ(ierr);        /*  beta <- z'*r       */
      }
      KSPCheckDot(ksp,beta);
    }
	
    i++;
	
	clock_gettime(CLOCK_REALTIME, &end_i);
	time_i += 1000000000L*(end_i.tv_sec - start_i.tv_sec) + (end_i.tv_nsec - start_i.tv_nsec);
	numi++;

	/* Dingwen */
		  /* Inject an error in P */
	  if((i==inj_itr)&&(flag3)&&(error_type==3))
	  {
		  srand (time(NULL));
		  VecScatter	ctx;
		  Vec			P_SEQ;
		  PetscScalar	*P_ARR;
		  VecScatterCreateToAll(P,&ctx,&P_SEQ);
		  VecScatterBegin(ctx,P,P_SEQ,INSERT_VALUES,SCATTER_FORWARD);
		  VecScatterEnd(ctx,P,P_SEQ,INSERT_VALUES,SCATTER_FORWARD);
		  VecGetArray(P_SEQ,&P_ARR);
		  pos1 = rand()/(RAND_MAX/n+1);
		  v		= P_ARR[pos1]*inj_times;
		  ierr	= VecSetValues(P,1,&pos1,&v,INSERT_VALUES);CHKERRQ(ierr);
		  VecDestroy(&P_SEQ);
		  VecScatterDestroy(&ctx);
		  VecAssemblyBegin(P);
		  VecAssemblyEnd(P);
		  PetscPrintf(MPI_COMM_WORLD,"Inject an error in P of position-%d at the end of iteration-%d\n", pos1,i);
		  flag3	= PETSC_FALSE;
	  }
	  
	  /* Inject two errors in P */
	  if((i==inj_itr)&&(flag4)&(error_type==4))
	  {
		  srand (time(NULL));
		  VecScatter	ctx;
		  Vec			P_SEQ;
		  PetscScalar	*P_ARR;
		  VecScatterCreateToAll(P,&ctx,&P_SEQ);
		  VecScatterBegin(ctx,P,P_SEQ,INSERT_VALUES,SCATTER_FORWARD);
		  VecScatterEnd(ctx,P,P_SEQ,INSERT_VALUES,SCATTER_FORWARD);
		  VecGetArray(P_SEQ,&P_ARR);
		  pos1 = rand()/(RAND_MAX/n+1);
		  v		= P_ARR[pos1]*inj_times;
		  ierr	= VecSetValues(P,1,&pos1,&v,INSERT_VALUES);CHKERRQ(ierr);
		  pos2 = rand()/(RAND_MAX/n+1);
		  v		= P_ARR[pos2]*inj_times;
		  ierr	= VecSetValues(P,1,&pos2,&v,INSERT_VALUES);CHKERRQ(ierr);
		  VecDestroy(&P_SEQ);
		  VecScatterDestroy(&ctx);
		  VecAssemblyBegin(P);
		  VecAssemblyEnd(P);
		  PetscPrintf(MPI_COMM_WORLD,"Inject two errors in P of position-%d and position-%d at the end of iteration-%d\n", pos1,pos2,i);
		  flag4	= PETSC_FALSE;
		}
		
		/* Inject an error in X*/
	  if((i==inj_itr)&&(flag5)&&(error_type==5))
	  {
		  srand (time(NULL));
		  VecScatter	ctx;
		  Vec			X_SEQ;
		  PetscScalar	*X_ARR;
		  VecScatterCreateToAll(X,&ctx,&X_SEQ);
		  VecScatterBegin(ctx,X,X_SEQ,INSERT_VALUES,SCATTER_FORWARD);
		  VecScatterEnd(ctx,X,X_SEQ,INSERT_VALUES,SCATTER_FORWARD);
		  VecGetArray(X_SEQ,&X_ARR);
		  pos1 = rand()/(RAND_MAX/n+1);
		  v		= X_ARR[pos1]*inj_times;
		  ierr	= VecSetValues(X,1,&pos1,&v,INSERT_VALUES);CHKERRQ(ierr);
		  VecDestroy(&X_SEQ);
		  VecScatterDestroy(&ctx);
		  VecAssemblyBegin(X);
		  VecAssemblyEnd(X);
		  PetscPrintf(MPI_COMM_WORLD,"Inject an error in X of position-%d at the end of iteration-%d\n", pos1,i);
		  flag5	= PETSC_FALSE;
	  }
	  
	  /* Inject two errors in X*/
	  if((i==inj_itr)&&(flag6)&&(error_type==6))
	  {
		  srand (time(NULL));
		  VecScatter	ctx;
		  Vec			X_SEQ;
		  PetscScalar	*X_ARR;
		  VecScatterCreateToAll(X,&ctx,&X_SEQ);
		  VecScatterBegin(ctx,X,X_SEQ,INSERT_VALUES,SCATTER_FORWARD);
		  VecScatterEnd(ctx,X,X_SEQ,INSERT_VALUES,SCATTER_FORWARD);
		  VecGetArray(X_SEQ,&X_ARR);
		  pos1 = rand()/(RAND_MAX/n+1);
		  v		= X_ARR[pos1]*inj_times;
		  ierr	= VecSetValues(X,1,&pos1,&v,INSERT_VALUES);CHKERRQ(ierr);
		  pos2 = rand()/(RAND_MAX/n+1);
		  v		= X_ARR[pos2]*inj_times;
		  ierr	= VecSetValues(X,1,&pos2,&v,INSERT_VALUES);CHKERRQ(ierr);
		  VecDestroy(&X_SEQ);
		  VecScatterDestroy(&ctx);
		  VecAssemblyBegin(X);
		  VecAssemblyEnd(X);
		  PetscPrintf(MPI_COMM_WORLD,"Inject two errors in X of position-%d and position-%d at the end of iteration-%d\n", pos1,pos2,i);
		  flag6	= PETSC_FALSE;
		}
		
		/* Inject an error in R */
	  if((i==inj_itr)&&(flag7)&&(error_type==7))
	  {
		  srand (time(NULL));
		  VecScatter	ctx;
		  Vec			R_SEQ;
		  PetscScalar	*R_ARR;
		  VecScatterCreateToAll(R,&ctx,&R_SEQ);
		  VecScatterBegin(ctx,R,R_SEQ,INSERT_VALUES,SCATTER_FORWARD);
		  VecScatterEnd(ctx,R,R_SEQ,INSERT_VALUES,SCATTER_FORWARD);
		  VecGetArray(R_SEQ,&R_ARR);
		  pos1 = rand()/(RAND_MAX/n+1);
		  v		= R_ARR[pos1]*inj_times;
		  ierr	= VecSetValues(R,1,&pos1,&v,INSERT_VALUES);CHKERRQ(ierr);
		  VecDestroy(&R_SEQ);
		  VecScatterDestroy(&ctx);
		  VecAssemblyBegin(R);
		  VecAssemblyEnd(R);
		  PetscPrintf(MPI_COMM_WORLD,"Inject an error in R of position-%d at the end of iteration-%d\n", pos1,i);
		  flag7	= PETSC_FALSE;
	  }
	  
	  /* Inject two errors */
	  if((i==inj_itr)&&(flag8)&&(error_type==8))
	  {
		  srand (time(NULL));
		  VecScatter	ctx;
		  Vec			R_SEQ;
		  PetscScalar	*R_ARR;
		  VecScatterCreateToAll(R,&ctx,&R_SEQ);
		  VecScatterBegin(ctx,R,R_SEQ,INSERT_VALUES,SCATTER_FORWARD);
		  VecScatterEnd(ctx,R,R_SEQ,INSERT_VALUES,SCATTER_FORWARD);
		  VecGetArray(R_SEQ,&R_ARR);
		  pos1 = rand()/(RAND_MAX/n+1);
		  v		= R_ARR[pos1]*inj_times;
		  ierr	= VecSetValues(R,1,&pos1,&v,INSERT_VALUES);CHKERRQ(ierr);
		  pos2 = rand()/(RAND_MAX/n+1);
		  v		= R_ARR[pos2]*inj_times;
		  ierr	= VecSetValues(R,1,&pos2,&v,INSERT_VALUES);CHKERRQ(ierr);
		  VecDestroy(&R_SEQ);
		  VecScatterDestroy(&ctx);
		  VecAssemblyBegin(R);
		  VecAssemblyEnd(R);
		  PetscPrintf(MPI_COMM_WORLD,"Inject two errors in R of position-%d and position-%d at the end of iteration-%d\n", pos1,pos2,i);
		  flag8	= PETSC_FALSE;
		}
		
		/* Inject an error in Z */
	  if((i==inj_itr)&&(flag9)&&(error_type==9))
	  {
		  srand (time(NULL));
		  VecScatter	ctx;
		  Vec			Z_SEQ;
		  PetscScalar	*Z_ARR;
		  VecScatterCreateToAll(Z,&ctx,&Z_SEQ);
		  VecScatterBegin(ctx,Z,Z_SEQ,INSERT_VALUES,SCATTER_FORWARD);
		  VecScatterEnd(ctx,Z,Z_SEQ,INSERT_VALUES,SCATTER_FORWARD);
		  VecGetArray(Z_SEQ,&Z_ARR);
		  pos1 = rand()/(RAND_MAX/n+1);
		  v		= Z_ARR[pos1]*inj_times;
		  ierr	= VecSetValues(Z,1,&pos1,&v,INSERT_VALUES);CHKERRQ(ierr);
		  VecDestroy(&Z_SEQ);
		  VecScatterDestroy(&ctx);
		  VecAssemblyBegin(Z);
		  VecAssemblyEnd(Z);
		  PetscPrintf(MPI_COMM_WORLD,"Inject an error in Z of position-%d at the end of iteration-%d\n", pos1,i);
		  flag9	= PETSC_FALSE;
	  }
	  
	  /* Inject two errors */
	  if((i==inj_itr)&&(flag10)&&(error_type==10))
	  {
		  srand (time(NULL));
		  VecScatter	ctx;
		  Vec			Z_SEQ;
		  PetscScalar	*Z_ARR;
		  VecScatterCreateToAll(Z,&ctx,&Z_SEQ);
		  VecScatterBegin(ctx,Z,Z_SEQ,INSERT_VALUES,SCATTER_FORWARD);
		  VecScatterEnd(ctx,Z,Z_SEQ,INSERT_VALUES,SCATTER_FORWARD);
		  VecGetArray(Z_SEQ,&Z_ARR);
		  pos1 = rand()/(RAND_MAX/n+1);
		  v		= Z_ARR[pos1]*inj_times;
		  ierr	= VecSetValues(Z,1,&pos1,&v,INSERT_VALUES);CHKERRQ(ierr);
		  pos2 = rand()/(RAND_MAX/n+1);
		  v		= Z_ARR[pos2]*inj_times;
		  ierr	= VecSetValues(Z,1,&pos2,&v,INSERT_VALUES);CHKERRQ(ierr);
		  VecDestroy(&Z_SEQ);
		  VecScatterDestroy(&ctx);
		  VecAssemblyBegin(Z);
		  VecAssemblyEnd(Z);
		  PetscPrintf(MPI_COMM_WORLD,"Inject two errors in Z of position-%d and position-%d at the end of iteration-%d\n", pos1,pos2,i);
		  flag10	= PETSC_FALSE;
		}
	/* Dingwen */
	
  } while (i<ksp->max_it);
  clock_gettime(CLOCK_REALTIME, &end);
  local_diff = 1000000000L*(end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec);
  MPI_Reduce(&local_diff, &global_diff, 1, MPI_LONG_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  PetscPrintf(MPI_COMM_WORLD,"Number of detections = %d\n", numd);
  PetscPrintf(MPI_COMM_WORLD,"Number of recoverys = %d\n", numr);
  PetscPrintf(MPI_COMM_WORLD,"Number of checkpoints = %d\n", numc);
  PetscPrintf(MPI_COMM_WORLD,"Number of iterations = %d\n", numi);
  PetscPrintf(MPI_COMM_WORLD,"Average time of each detection = %lf nanoseconds\n", (double) (time_d)/numd);
  PetscPrintf(MPI_COMM_WORLD,"Average time of each recovery = %lf nanoseconds\n", (double) (time_r)/numr);
  PetscPrintf(MPI_COMM_WORLD,"Average time of each checkpoint = %lf nanoseconds\n", (double) (time_c)/numc);
  PetscPrintf(MPI_COMM_WORLD,"Average time of each iteration = %lf nanoseconds\n", (double) (time_i)/numi);
  PetscPrintf(MPI_COMM_WORLD,"Elapsed time of main loop = %lf nanoseconds\n", (double) (global_diff)/size);
  PetscPrintf(MPI_COMM_WORLD,"Number of iterations without rollback = %d\n", i+1);
  }
  
  if (solver_type==2)
  {
  cg            = (KSP_CG*)ksp->data;
  eigs          = ksp->calc_sings;
  stored_max_it = ksp->max_it;
  X             = ksp->vec_sol;
  B             = ksp->vec_rhs;
  R             = ksp->work[0];
  Z             = ksp->work[1];
  P             = ksp->work[2];
  /* Dingwen */
  CKPX			= ksp->work[3];
  CKPP			= ksp->work[4];
  CKSAmat		= ksp->work[5];
  C1			= ksp->work[6];
  /* Dingwen */
  
  if (cg->singlereduction) {
    S = ksp->work[8];
    W = ksp->work[9];
  } else {
    S = 0;                      /* unused */
    W = Z;
  }
    
  if (eigs) {e = cg->e; d = cg->d; e[0] = 0.0; }
  ierr = PCGetOperators(ksp->pc,&Amat,&Pmat);CHKERRQ(ierr);
  
  ksp->its = 0;
  if (!ksp->guess_zero) {
    ierr = KSP_MatMult(ksp,Amat,X,R);CHKERRQ(ierr);            /*     r <- b - Ax     */
    ierr = VecAYPX(R,-1.0,B);CHKERRQ(ierr);
  } else {
    ierr = VecCopy(B,R);CHKERRQ(ierr);                         /*     r <- b (x is 0) */
  }
  

  /* Dingwen */	
  /* checksum coefficients initialization */
  PetscInt n;
  PetscInt *index;
  PetscScalar *v1;
  ierr = VecGetSize(B,&n);
  v1 	= (PetscScalar *)malloc(n*sizeof(PetscScalar));
  index	= (PetscInt *)malloc(n*sizeof(PetscInt));
  for (i=0; i<n; i++)
  {
	  index[i] = i;
	  v1[i] = 1.0;
  }
  ierr	= VecSetValues(C1,n,index,v1,INSERT_VALUES);CHKERRQ(ierr);	
  /* Dingwen */

  switch (ksp->normtype) {
  case KSP_NORM_PRECONDITIONED:
    ierr = KSP_PCApply(ksp,R,Z);CHKERRQ(ierr);                   /*     z <- Br         */
    ierr = VecNorm(Z,NORM_2,&dp);CHKERRQ(ierr);                /*    dp <- z'*z = e'*A'*B'*B*A'*e'     */
    break;
  case KSP_NORM_UNPRECONDITIONED:
    ierr = VecNorm(R,NORM_2,&dp);CHKERRQ(ierr);                /*    dp <- r'*r = e'*A'*A*e            */
    break;
  case KSP_NORM_NATURAL:
    ierr = KSP_PCApply(ksp,R,Z);CHKERRQ(ierr);                   /*     z <- Br         */
    if (cg->singlereduction) {
      ierr = KSP_MatMult(ksp,Amat,Z,S);CHKERRQ(ierr);
      ierr = VecXDot(Z,S,&delta);CHKERRQ(ierr);
	}
    ierr = VecXDot(Z,R,&beta);CHKERRQ(ierr);                     /*  beta <- z'*r       */
    KSPCheckDot(ksp,beta);
    dp = PetscSqrtReal(PetscAbsScalar(beta));                           /*    dp <- r'*z = r'*B*r = e'*A'*B*A*e */
    break;
  case KSP_NORM_NONE:
    dp = 0.0;
    break;
  default: SETERRQ1(PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"%s",KSPNormTypes[ksp->normtype]);
  }
  
  ierr       = KSPLogResidualHistory(ksp,dp);CHKERRQ(ierr);
  ierr       = KSPMonitor(ksp,0,dp);CHKERRQ(ierr);
  ksp->rnorm = dp;

  ierr = (*ksp->converged)(ksp,0,dp,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);      /* test for convergence */
  if (ksp->reason) PetscFunctionReturn(0);

  if (ksp->normtype != KSP_NORM_PRECONDITIONED && (ksp->normtype != KSP_NORM_NATURAL)) {
    ierr = KSP_PCApply(ksp,R,Z);CHKERRQ(ierr);                   /*     z <- Br         */
  }
  if (ksp->normtype != KSP_NORM_NATURAL) {
    if (cg->singlereduction) {
      ierr = KSP_MatMult(ksp,Amat,Z,S);CHKERRQ(ierr);
      ierr = VecXDot(Z,S,&delta);CHKERRQ(ierr);
    }
    ierr = VecXDot(Z,R,&beta);CHKERRQ(ierr);         /*  beta <- z'*r       */
    KSPCheckDot(ksp,beta);
  }

  /* Dingwen */
  /* Checksum Initialization */
  ierr = VecXDot(C1,X,&CKSX);CHKERRQ(ierr);						/* Compute the initial checksum(X) */ 
  ierr = VecXDot(C1,W,&CKSW);CHKERRQ(ierr);						/* Compute the initial checksum(W) */
  ierr = VecXDot(C1,R,&CKSR);CHKERRQ(ierr);						/* Compute the initial checksum(R) */
  ierr = VecXDot(C1,Z,&CKSZ);CHKERRQ(ierr);						/* Compute the initial checksum(Z) */
  ierr = KSP_MatMultTranspose(ksp,Amat,C1,CKSAmat);CHKERRQ(ierr);
  /* Dingwen */
  
  struct timespec start, end;
  struct timespec start_MVM, end_MVM;
  long long int local_diff, global_diff;
  long long int time_MVM = 0;
  int num_MVM = 0;
  
  clock_gettime(CLOCK_REALTIME, &start);
  i = 0;
  do {
	  if ((i>0)&&(i%(itv_c*itv_d) == 0))
		{
			ierr = VecCopy(X,CKPX);CHKERRQ(ierr);
			ierr = VecCopy(P,CKPP);CHKERRQ(ierr);
			CKPbetaold = betaold;
			CKPi = i;
		}
	  ksp->its++;
	  if (beta == 0.0) {
      ksp->reason = KSP_CONVERGED_ATOL;
      ierr        = PetscInfo(ksp,"converged due to beta = 0\n");CHKERRQ(ierr);
      break;
#if !defined(PETSC_USE_COMPLEX)
    } else if ((i > 0) && (beta*betaold < 0.0)) {
      ksp->reason = KSP_DIVERGED_INDEFINITE_PC;
      ierr        = PetscInfo(ksp,"diverging due to indefinite preconditioner\n");CHKERRQ(ierr);
      break;
#endif
    }
    if (!i) {
      ierr = VecCopy(Z,P);CHKERRQ(ierr);         /*     p <- z          */
      b    = 0.0;
	  /* Dingwen */
	  ierr = VecXDot(C1,P, &CKSP);CHKERRQ(ierr);  				/* Compute the initial checksum(P) */
	  /* Dingwen */
    } else {
      b = beta/betaold;
      if (eigs) {
        if (ksp->max_it != stored_max_it) SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"Can not change maxit AND calculate eigenvalues");
        e[i] = PetscSqrtReal(PetscAbsScalar(b))/a;
      }
      ierr = VecAYPX(P,b,Z);CHKERRQ(ierr);    /*     p <- z + b* p   */
	  /* Dingwen */
	  CKSP = CKSZ + b*CKSP;										/* Update checksum(P) = checksum(Z) + b*checksum(P); */
	  
	  ierr = VecXDot(C1,P, &sumP);CHKERRQ(ierr);
	  if (PetscAbsScalar(CKSP-sumP)/n > 1.0e-6)				/* Check checksum(P) = sum(P) */
	  {
		  PetscPrintf(MPI_COMM_WORLD,"Recovery start...\n");
		  PetscPrintf(MPI_COMM_WORLD,"Rollback from iteration-%d to iteration-%d\n",i,CKPi);
		  betaold = CKPbetaold;										/* Recovery scalar betaold by checkpoint*/
		  i = CKPi;													/* Recovery integer i by checkpoint */
		  ierr = VecCopy(CKPP,P);CHKERRQ(ierr);						/* Recovery vector P from checkpoint */
		  ierr = VecXDot(C1,P,&CKSP);CHKERRQ(ierr);					/* Recovery checksum(P) by P */ 
		  ierr = KSP_MatMult(ksp,Amat,P,W);CHKERRQ(ierr);			/* Recovery vector W by P */
		  ierr = VecXDot(P,W,&dpi);CHKERRQ(ierr);					/* Recovery scalar dpi by P and W */
		  ierr = VecCopy(CKPX,X);CHKERRQ(ierr);						/* Recovery vector X from checkpoint */
		  ierr = VecXDot(C1,X,&CKSX);CHKERRQ(ierr);					/* Recovery checksum(X) by X */ 
		  ierr = KSP_MatMult(ksp,Amat,X,R);CHKERRQ(ierr);			/* Recovery vector R by X */
		  ierr = VecAYPX(R,-1.0,B);CHKERRQ(ierr);
		  ierr = VecXDot(C1,R,&CKSR);CHKERRQ(ierr);					/* Recovery checksum(R) by R */
		  ierr = KSP_PCApply(ksp,R,Z);CHKERRQ(ierr);				/* Recovery vector Z by R */
		  ierr = VecXDot(C1,Z,&CKSZ);CHKERRQ(ierr);					/* Recovery checksum(Z) by Z */
		  ierr = VecXDot(Z,R,&beta);CHKERRQ(ierr);					/* Recovery scalar beta by Z and R */
		  PetscPrintf(MPI_COMM_WORLD,"Recovery end.\n");

		  /* Recover the calculations from iteration begining to checking */
		  b = beta/betaold;
		  ierr = VecAYPX(P,b,Z);CHKERRQ(ierr);    /*     p <- z + b* p   */	
		  CKSP = CKSZ + b*CKSP;					/* Update checksum(P) = checksum(Z) + b*checksum(P); */
	  }
	  /* Dingwen */
    }
    dpiold = dpi;
    if (!cg->singlereduction || !i) {
	
	  clock_gettime(CLOCK_REALTIME, &start_MVM);
	  ierr = KSP_MatMult(ksp,Amat,P,W);CHKERRQ(ierr);          /*     w <- Ap         */	/* MVM */
	  clock_gettime(CLOCK_REALTIME, &end_MVM);
	  time_MVM += 1000000000L*(end_MVM.tv_sec - start_MVM.tv_sec) + (end_MVM.tv_nsec - start_MVM	.tv_nsec);
	  num_MVM++;
  
	  /* Dingwen */
	  ierr = VecXDot(CKSAmat,P, &CKSW);CHKERRQ(ierr);
	  
	  /* Inject an error in W */
	  if((i==inj_itr)&&(flag1)&&(error_type==1))
	  {
		  srand (time(NULL));
		  VecScatter	ctx;
		  Vec			W_SEQ;
		  PetscScalar	*W_ARR;
		  VecScatterCreateToAll(W,&ctx,&W_SEQ);
		  VecScatterBegin(ctx,W,W_SEQ,INSERT_VALUES,SCATTER_FORWARD);
		  VecScatterEnd(ctx,W,W_SEQ,INSERT_VALUES,SCATTER_FORWARD);
		  VecGetArray(W_SEQ,&W_ARR);
		  pos1 = rand()/(RAND_MAX/n+1);
		  v		= W_ARR[pos1]*inj_times;
		  ierr	= VecSetValues(W,1,&pos1,&v,INSERT_VALUES);CHKERRQ(ierr);
		  VecDestroy(&W_SEQ);
		  VecScatterDestroy(&ctx);
		  VecAssemblyBegin(W);
		  VecAssemblyEnd(W);
		  PetscPrintf(MPI_COMM_WORLD,"Inject an error in W of position-%d after MVM at iteration-%d\n", pos1,i);
		  flag1	= PETSC_FALSE;
	  }
	  
	  /* Inject two errors in W */
	  if((i==inj_itr)&&(flag2)&(error_type==2))
	  {
		  srand (time(NULL));
		  VecScatter	ctx;
		  Vec			W_SEQ;
		  PetscScalar	*W_ARR;
		  VecScatterCreateToAll(W,&ctx,&W_SEQ);
		  VecScatterBegin(ctx,W,W_SEQ,INSERT_VALUES,SCATTER_FORWARD);
		  VecScatterEnd(ctx,W,W_SEQ,INSERT_VALUES,SCATTER_FORWARD);
		  VecGetArray(W_SEQ,&W_ARR);
		  pos1 = rand()/(RAND_MAX/n+1);
		  v		= W_ARR[pos1]*inj_times;
		  ierr	= VecSetValues(W,1,&pos1,&v,INSERT_VALUES);CHKERRQ(ierr);
		  pos2 = rand()/(RAND_MAX/n+1);
		  v		= W_ARR[pos2]*inj_times;
		  ierr	= VecSetValues(W,1,&pos2,&v,INSERT_VALUES);CHKERRQ(ierr);
		  VecDestroy(&W_SEQ);
		  VecScatterDestroy(&ctx);
		  VecAssemblyBegin(W);
		  VecAssemblyEnd(W);
		  PetscPrintf(MPI_COMM_WORLD,"Inject two errors in W of position-%d and position-%d after MVM at iteration-%d\n", pos1,pos2,i);
		  flag2	= PETSC_FALSE;
		}
		
		ierr = VecXDot(C1,W, &sumW);CHKERRQ(ierr);
	  if (PetscAbsScalar(CKSW-sumW)/ n > 1.0e-6)				/* Check checksum(W) = sum(W) */
	  {
		  PetscPrintf(MPI_COMM_WORLD,"Recovery start...\n");
		  PetscPrintf(MPI_COMM_WORLD,"Rollback from iteration-%d to iteration-%d\n",i,CKPi);
		  betaold = CKPbetaold;										/* Recovery scalar betaold by checkpoint*/
		  i = CKPi;													/* Recovery integer i by checkpoint */
		  ierr = VecCopy(CKPP,P);CHKERRQ(ierr);						/* Recovery vector P from checkpoint */
		  ierr = VecXDot(C1,P,&CKSP);CHKERRQ(ierr);					/* Recovery checksum(P) by P */ 
		  ierr = KSP_MatMult(ksp,Amat,P,W);CHKERRQ(ierr);			/* Recovery vector W by P */
		  ierr = VecXDot(P,W,&dpi);CHKERRQ(ierr);					/* Recovery scalar dpi by P and W */
		  ierr = VecCopy(CKPX,X);CHKERRQ(ierr);						/* Recovery vector X from checkpoint */
		  ierr = VecXDot(C1,X,&CKSX);CHKERRQ(ierr);					/* Recovery checksum(X) by X */ 
		  ierr = KSP_MatMult(ksp,Amat,X,R);CHKERRQ(ierr);			/* Recovery vector R by X */
		  ierr = VecAYPX(R,-1.0,B);CHKERRQ(ierr);
		  ierr = VecXDot(C1,R,&CKSR);CHKERRQ(ierr);					/* Recovery checksum(R) by R */
		  ierr = KSP_PCApply(ksp,R,Z);CHKERRQ(ierr);				/* Recovery vector Z by R */
		  ierr = VecXDot(C1,Z,&CKSZ);CHKERRQ(ierr);					/* Recovery checksum(Z) by Z */
		  ierr = VecXDot(Z,R,&beta);CHKERRQ(ierr);					/* Recovery scalar beta by Z and R */
		  PetscPrintf(MPI_COMM_WORLD,"Recovery end.\n");

		  /* Recover the calculations from iteration begining to checking */
		  b = beta/betaold;
		  ierr = VecAYPX(P,b,Z);CHKERRQ(ierr);    /*     p <- z + b* p   */
		  CKSP = CKSZ + b*CKSP;										/* Update checksum(P) = checksum(Z) + b*checksum(P); */
		  dpiold = dpi;
		  ierr = KSP_MatMult(ksp,Amat,P,W);CHKERRQ(ierr);          /*     w <- Ap         */	/* MVM */
		  ierr = VecXDot(CKSAmat,P, &CKSW);CHKERRQ(ierr);
	  }	  
	  /* Dingwen */
	  
	  ierr = VecXDot(P,W,&dpi);CHKERRQ(ierr);                  /*     dpi <- p'w     */	  
	  
    } else {
      ierr = VecAYPX(W,beta/betaold,S);CHKERRQ(ierr);                  /*     w <- Ap         */
      dpi  = delta - beta*beta*dpiold/(betaold*betaold);             /*     dpi <- p'w     */
	}
    betaold = beta;
    KSPCheckDot(ksp,beta);

    if ((dpi == 0.0) || ((i > 0) && (PetscRealPart(dpi*dpiold) <= 0.0))) {
      ksp->reason = KSP_DIVERGED_INDEFINITE_MAT;
      ierr        = PetscInfo(ksp,"diverging due to indefinite or negative definite matrix\n");CHKERRQ(ierr);
      break;
    }
    a = beta/dpi;                                 /*     a = beta/p'w   */
    if (eigs) d[i] = PetscSqrtReal(PetscAbsScalar(b))*e[i] + 1.0/a;
    ierr = VecAXPY(X,a,P);CHKERRQ(ierr);          /*     x <- x + ap     */
	/* Dingwen */
	CKSX = CKSX + a*CKSP;									/* Update checksum(X) = checksum(X) + a*checksum(P); */
	
	ierr = VecXDot(C1,X, &sumX);CHKERRQ(ierr);
	  if (PetscAbsScalar(CKSX-sumX)/n > 1.0e-6)			/* Check checksum(X) = sum(X) */
	  {
		  PetscPrintf(MPI_COMM_WORLD,"Recovery start...\n");
		  PetscPrintf(MPI_COMM_WORLD,"Rollback from iteration-%d to iteration-%d\n",i,CKPi);
		  betaold = CKPbetaold;										/* Recovery scalar betaold by checkpoint*/
		  i = CKPi;													/* Recovery integer i by checkpoint */
		  ierr = VecCopy(CKPP,P);CHKERRQ(ierr);						/* Recovery vector P from checkpoint */
		  ierr = VecXDot(C1,P,&CKSP);CHKERRQ(ierr);					/* Recovery checksum(P) by P */ 
		  ierr = KSP_MatMult(ksp,Amat,P,W);CHKERRQ(ierr);			/* Recovery vector W by P */
		  ierr = VecXDot(P,W,&dpi);CHKERRQ(ierr);					/* Recovery scalar dpi by P and W */
		  ierr = VecCopy(CKPX,X);CHKERRQ(ierr);						/* Recovery vector X from checkpoint */
		  ierr = VecXDot(C1,X,&CKSX);CHKERRQ(ierr);					/* Recovery checksum(X) by X */ 
		  ierr = KSP_MatMult(ksp,Amat,X,R);CHKERRQ(ierr);			/* Recovery vector R by X */
		  ierr = VecAYPX(R,-1.0,B);CHKERRQ(ierr);
		  ierr = VecXDot(C1,R,&CKSR);CHKERRQ(ierr);					/* Recovery checksum(R) by R */
		  ierr = KSP_PCApply(ksp,R,Z);CHKERRQ(ierr);				/* Recovery vector Z by R */
		  ierr = VecXDot(C1,Z,&CKSZ);CHKERRQ(ierr);					/* Recovery checksum(Z) by Z */
		  ierr = VecXDot(Z,R,&beta);CHKERRQ(ierr);					/* Recovery scalar beta by Z and R */
		  PetscPrintf(MPI_COMM_WORLD,"Recovery end.\n");

		  /* Recover the calculations from iteration begining to checking */
		  b = beta/betaold;
		  ierr = VecAYPX(P,b,Z);CHKERRQ(ierr);    /*     p <- z + b* p   */
		  CKSP = CKSZ + b*CKSP;										/* Update checksum(P) = checksum(Z) + b*checksum(P); */
		  dpiold = dpi;
		  ierr = KSP_MatMult(ksp,Amat,P,W);CHKERRQ(ierr);          /*     w <- Ap         */	/* MVM */
		  ierr = VecXDot(CKSAmat,P, &CKSW);CHKERRQ(ierr);
		  ierr = VecXDot(P,W,&dpi);CHKERRQ(ierr);                  /*     dpi <- p'w     */	  
		  betaold = beta;
		  a = beta/dpi;                                 /*     a = beta/p'w   */
		  ierr = VecAXPY(X,a,P);CHKERRQ(ierr);          /*     x <- x + ap     */
		  CKSX = CKSX + a*CKSP;									/* Update checksum(X) = checksum(X) + a*checksum(P); */
	  }

	/* Dingwen */
    
	ierr = VecAXPY(R,-a,W);CHKERRQ(ierr);                      /*     r <- r - aw    */

	/* Dingwen */
	CKSR = CKSR - a*CKSW;									/* Update checksum(R) = checksum(R) - a*checksum(W); */
	
	ierr = VecXDot(C1,R, &sumR);CHKERRQ(ierr);
	if (PetscAbsScalar(CKSR-sumR)/n > 1.0e-6)			/* Check checksum(R) = sum(R) */
	{
		  PetscPrintf(MPI_COMM_WORLD,"Recovery start...\n");
		  PetscPrintf(MPI_COMM_WORLD,"Rollback from iteration-%d to iteration-%d\n",i,CKPi);
		  betaold = CKPbetaold;										/* Recovery scalar betaold by checkpoint*/
		  i = CKPi;													/* Recovery integer i by checkpoint */
		  ierr = VecCopy(CKPP,P);CHKERRQ(ierr);						/* Recovery vector P from checkpoint */
		  ierr = VecXDot(C1,P,&CKSP);CHKERRQ(ierr);					/* Recovery checksum(P) by P */ 
		  ierr = KSP_MatMult(ksp,Amat,P,W);CHKERRQ(ierr);			/* Recovery vector W by P */
		  ierr = VecXDot(P,W,&dpi);CHKERRQ(ierr);					/* Recovery scalar dpi by P and W */
		  ierr = VecCopy(CKPX,X);CHKERRQ(ierr);						/* Recovery vector X from checkpoint */
		  ierr = VecXDot(C1,X,&CKSX);CHKERRQ(ierr);					/* Recovery checksum(X) by X */ 
		  ierr = KSP_MatMult(ksp,Amat,X,R);CHKERRQ(ierr);			/* Recovery vector R by X */
		  ierr = VecAYPX(R,-1.0,B);CHKERRQ(ierr);
		  ierr = VecXDot(C1,R,&CKSR);CHKERRQ(ierr);					/* Recovery checksum(R) by R */
		  ierr = KSP_PCApply(ksp,R,Z);CHKERRQ(ierr);				/* Recovery vector Z by R */
		  ierr = VecXDot(C1,Z,&CKSZ);CHKERRQ(ierr);					/* Recovery checksum(Z) by Z */
		  ierr = VecXDot(Z,R,&beta);CHKERRQ(ierr);					/* Recovery scalar beta by Z and R */
		  PetscPrintf(MPI_COMM_WORLD,"Recovery end.\n");

		  /* Recover the calculations from iteration begining to checking */
		  b = beta/betaold;
		  ierr = VecAYPX(P,b,Z);CHKERRQ(ierr);    /*     p <- z + b* p   */
		  CKSP = CKSZ + b*CKSP;										/* Update checksum(P) = checksum(Z) + b*checksum(P); */
		  dpiold = dpi;
		  ierr = KSP_MatMult(ksp,Amat,P,W);CHKERRQ(ierr);          /*     w <- Ap         */	/* MVM */
		  ierr = VecXDot(CKSAmat,P, &CKSW);CHKERRQ(ierr);
		  ierr = VecXDot(P,W,&dpi);CHKERRQ(ierr);                  /*     dpi <- p'w     */	  
		  betaold = beta;
		  a = beta/dpi;                                 /*     a = beta/p'w   */
		  ierr = VecAXPY(X,a,P);CHKERRQ(ierr);          /*     x <- x + ap     */
		  CKSX = CKSX + a*CKSP;									/* Update checksum(X) = checksum(X) + a*checksum(P); */
		  ierr = VecAXPY(R,-a,W);CHKERRQ(ierr);                      /*     r <- r - aw    */
		  CKSR = CKSR - a*CKSW;									/* Update checksum(R) = checksum(R) - a*checksum(W); */
	}
	/* Dingwen */
	
	if (ksp->normtype == KSP_NORM_PRECONDITIONED && ksp->chknorm < i+2) {      
	  ierr = KSP_PCApply(ksp,R,Z);CHKERRQ(ierr);               /*     z <- Br         */
	  
	  /* Dingwen */
	  ierr = VecXDot(C1,Z, &CKSZ);CHKERRQ(ierr);				/* Update checksum(Z) */
	  /* Dingwen */
	  
	  if (cg->singlereduction) {
        ierr = KSP_MatMult(ksp,Amat,Z,S);CHKERRQ(ierr);			/* MVM */
      }
      ierr = VecNorm(Z,NORM_2,&dp);CHKERRQ(ierr);              /*    dp <- z'*z       */
    } else if (ksp->normtype == KSP_NORM_UNPRECONDITIONED && ksp->chknorm < i+2) {
      ierr = VecNorm(R,NORM_2,&dp);CHKERRQ(ierr);              /*    dp <- r'*r       */
    } else if (ksp->normtype == KSP_NORM_NATURAL) {
      ierr = KSP_PCApply(ksp,R,Z);CHKERRQ(ierr);               /*     z <- Br         */
 	  
	  /* Dingwen */
	  ierr = VecXDot(C1,Z, &CKSZ);CHKERRQ(ierr);				/* Update checksum(Z) */
	  /* Dingwen */
	  
	  if (cg->singlereduction) {
        PetscScalar tmp[2];
        Vec         vecs[2];
        vecs[0] = S; vecs[1] = R;
        ierr    = KSP_MatMult(ksp,Amat,Z,S);CHKERRQ(ierr);
        ierr  = VecMDot(Z,2,vecs,tmp);CHKERRQ(ierr);
        delta = tmp[0]; beta = tmp[1];
      } else {
        ierr = VecXDot(Z,R,&beta);CHKERRQ(ierr);     /*  beta <- r'*z       */
      }
      KSPCheckDot(ksp,beta);
      dp = PetscSqrtReal(PetscAbsScalar(beta));
    } else {
      dp = 0.0;
    }
	  
    ksp->rnorm = dp;
    CHKERRQ(ierr);KSPLogResidualHistory(ksp,dp);CHKERRQ(ierr);
    ierr = KSPMonitor(ksp,i+1,dp);CHKERRQ(ierr);
    ierr = (*ksp->converged)(ksp,i+1,dp,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
    if (ksp->reason) break;

    if ((ksp->normtype != KSP_NORM_PRECONDITIONED && (ksp->normtype != KSP_NORM_NATURAL)) || (ksp->chknorm >= i+2)) {
      ierr = KSP_PCApply(ksp,R,Z);CHKERRQ(ierr);                   /*     z <- Br         */
	  
	  /* Dingwen */
	  ierr = VecXDot(C1,Z, &CKSZ);CHKERRQ(ierr);				/* Update checksum(Z) */
	  /* Dingwen */
      
	  if (cg->singlereduction) {
        ierr = KSP_MatMult(ksp,Amat,Z,S);CHKERRQ(ierr);
      }
    }
		  
    if ((ksp->normtype != KSP_NORM_NATURAL) || (ksp->chknorm >= i+2)) {
      if (cg->singlereduction) {
        PetscScalar tmp[2];
        Vec         vecs[2];
        vecs[0] = S; vecs[1] = R;
        ierr  = VecMDot(Z,2,vecs,tmp);CHKERRQ(ierr);
        delta = tmp[0]; beta = tmp[1];
      } else {
        ierr = VecXDot(Z,R,&beta);CHKERRQ(ierr);        /*  beta <- z'*r       */
      }
      KSPCheckDot(ksp,beta);
    }
	
    i++;
	
	/* Dingwen */
	/* Inject error */
	  /* Inject an error in P */
	  if((i==inj_itr)&&(flag3)&&(error_type==3))
	  {
		  srand (time(NULL));
		  VecScatter	ctx;
		  Vec			P_SEQ;
		  PetscScalar	*P_ARR;
		  VecScatterCreateToAll(P,&ctx,&P_SEQ);
		  VecScatterBegin(ctx,P,P_SEQ,INSERT_VALUES,SCATTER_FORWARD);
		  VecScatterEnd(ctx,P,P_SEQ,INSERT_VALUES,SCATTER_FORWARD);
		  VecGetArray(P_SEQ,&P_ARR);
		  pos1 = rand()/(RAND_MAX/n+1);
		  v		= P_ARR[pos1]*inj_times;
		  ierr	= VecSetValues(P,1,&pos1,&v,INSERT_VALUES);CHKERRQ(ierr);
		  VecDestroy(&P_SEQ);
		  VecScatterDestroy(&ctx);
		  VecAssemblyBegin(P);
		  VecAssemblyEnd(P);
		  PetscPrintf(MPI_COMM_WORLD,"Inject an error in P of position-%d at the end of iteration-%d\n", pos1,i);
		  flag3	= PETSC_FALSE;
	  }
	  
	  /* Inject two errors in P */
	  if((i==inj_itr)&&(flag4)&(error_type==4))
	  {
		  srand (time(NULL));
		  VecScatter	ctx;
		  Vec			P_SEQ;
		  PetscScalar	*P_ARR;
		  VecScatterCreateToAll(P,&ctx,&P_SEQ);
		  VecScatterBegin(ctx,P,P_SEQ,INSERT_VALUES,SCATTER_FORWARD);
		  VecScatterEnd(ctx,P,P_SEQ,INSERT_VALUES,SCATTER_FORWARD);
		  VecGetArray(P_SEQ,&P_ARR);
		  pos1 = rand()/(RAND_MAX/n+1);
		  v		= P_ARR[pos1]*inj_times;
		  ierr	= VecSetValues(P,1,&pos1,&v,INSERT_VALUES);CHKERRQ(ierr);
		  pos2 = rand()/(RAND_MAX/n+1);
		  v		= P_ARR[pos2]*inj_times;
		  ierr	= VecSetValues(P,1,&pos2,&v,INSERT_VALUES);CHKERRQ(ierr);
		  VecDestroy(&P_SEQ);
		  VecScatterDestroy(&ctx);
		  VecAssemblyBegin(P);
		  VecAssemblyEnd(P);
		  PetscPrintf(MPI_COMM_WORLD,"Inject two errors in P of position-%d and position-%d at the end of iteration-%d\n", pos1,pos2,i);
		  flag4	= PETSC_FALSE;
		}
		
		/* Inject an error in X*/
	  if((i==inj_itr)&&(flag5)&&(error_type==5))
	  {
		  srand (time(NULL));
		  VecScatter	ctx;
		  Vec			X_SEQ;
		  PetscScalar	*X_ARR;
		  VecScatterCreateToAll(X,&ctx,&X_SEQ);
		  VecScatterBegin(ctx,X,X_SEQ,INSERT_VALUES,SCATTER_FORWARD);
		  VecScatterEnd(ctx,X,X_SEQ,INSERT_VALUES,SCATTER_FORWARD);
		  VecGetArray(X_SEQ,&X_ARR);
		  pos1 = rand()/(RAND_MAX/n+1);
		  v		= X_ARR[pos1]*inj_times;
		  ierr	= VecSetValues(X,1,&pos1,&v,INSERT_VALUES);CHKERRQ(ierr);
		  VecDestroy(&X_SEQ);
		  VecScatterDestroy(&ctx);
		  VecAssemblyBegin(X);
		  VecAssemblyEnd(X);
		  PetscPrintf(MPI_COMM_WORLD,"Inject an error in X of position-%d at the end of iteration-%d\n", pos1,i);
		  flag5	= PETSC_FALSE;
	  }
	  
	  /* Inject two errors in X*/
	  if((i==inj_itr)&&(flag6)&&(error_type==6))
	  {
		  srand (time(NULL));
		  VecScatter	ctx;
		  Vec			X_SEQ;
		  PetscScalar	*X_ARR;
		  VecScatterCreateToAll(X,&ctx,&X_SEQ);
		  VecScatterBegin(ctx,X,X_SEQ,INSERT_VALUES,SCATTER_FORWARD);
		  VecScatterEnd(ctx,X,X_SEQ,INSERT_VALUES,SCATTER_FORWARD);
		  VecGetArray(X_SEQ,&X_ARR);
		  pos1 = rand()/(RAND_MAX/n+1);
		  v		= X_ARR[pos1]*inj_times;
		  ierr	= VecSetValues(X,1,&pos1,&v,INSERT_VALUES);CHKERRQ(ierr);
		  pos2 = rand()/(RAND_MAX/n+1);
		  v		= X_ARR[pos2]*inj_times;
		  ierr	= VecSetValues(X,1,&pos2,&v,INSERT_VALUES);CHKERRQ(ierr);
		  VecDestroy(&X_SEQ);
		  VecScatterDestroy(&ctx);
		  VecAssemblyBegin(X);
		  VecAssemblyEnd(X);
		  PetscPrintf(MPI_COMM_WORLD,"Inject two errors in X of position-%d and position-%d at the end of iteration-%d\n", pos1,pos2,i);
		  flag6	= PETSC_FALSE;
		}
		
		/* Inject an error in R */
	  if((i==inj_itr)&&(flag7)&&(error_type==7))
	  {
		  srand (time(NULL));
		  VecScatter	ctx;
		  Vec			R_SEQ;
		  PetscScalar	*R_ARR;
		  VecScatterCreateToAll(R,&ctx,&R_SEQ);
		  VecScatterBegin(ctx,R,R_SEQ,INSERT_VALUES,SCATTER_FORWARD);
		  VecScatterEnd(ctx,R,R_SEQ,INSERT_VALUES,SCATTER_FORWARD);
		  VecGetArray(R_SEQ,&R_ARR);
		  pos1 = rand()/(RAND_MAX/n+1);
		  v		= R_ARR[pos1]*inj_times;
		  ierr	= VecSetValues(R,1,&pos1,&v,INSERT_VALUES);CHKERRQ(ierr);
		  VecDestroy(&R_SEQ);
		  VecScatterDestroy(&ctx);
		  VecAssemblyBegin(R);
		  VecAssemblyEnd(R);
		  PetscPrintf(MPI_COMM_WORLD,"Inject an error in R of position-%d at the end of iteration-%d\n", pos1,i);
		  flag7	= PETSC_FALSE;
	  }
	  
	  /* Inject two errors */
	  if((i==inj_itr)&&(flag8)&&(error_type==8))
	  {
		  srand (time(NULL));
		  VecScatter	ctx;
		  Vec			R_SEQ;
		  PetscScalar	*R_ARR;
		  VecScatterCreateToAll(R,&ctx,&R_SEQ);
		  VecScatterBegin(ctx,R,R_SEQ,INSERT_VALUES,SCATTER_FORWARD);
		  VecScatterEnd(ctx,R,R_SEQ,INSERT_VALUES,SCATTER_FORWARD);
		  VecGetArray(R_SEQ,&R_ARR);
		  pos1 = rand()/(RAND_MAX/n+1);
		  v		= R_ARR[pos1]*inj_times;
		  ierr	= VecSetValues(R,1,&pos1,&v,INSERT_VALUES);CHKERRQ(ierr);
		  pos2 = rand()/(RAND_MAX/n+1);
		  v		= R_ARR[pos2]*inj_times;
		  ierr	= VecSetValues(R,1,&pos2,&v,INSERT_VALUES);CHKERRQ(ierr);
		  VecDestroy(&R_SEQ);
		  VecScatterDestroy(&ctx);
		  VecAssemblyBegin(R);
		  VecAssemblyEnd(R);
		  PetscPrintf(MPI_COMM_WORLD,"Inject two errors in R of position-%d and position-%d at the end of iteration-%d\n", pos1,pos2,i);
		  flag8	= PETSC_FALSE;
		}
		
		/* Inject an error in Z */
	  if((i==inj_itr)&&(flag9)&&(error_type==9))
	  {
		  srand (time(NULL));
		  VecScatter	ctx;
		  Vec			Z_SEQ;
		  PetscScalar	*Z_ARR;
		  VecScatterCreateToAll(Z,&ctx,&Z_SEQ);
		  VecScatterBegin(ctx,Z,Z_SEQ,INSERT_VALUES,SCATTER_FORWARD);
		  VecScatterEnd(ctx,Z,Z_SEQ,INSERT_VALUES,SCATTER_FORWARD);
		  VecGetArray(Z_SEQ,&Z_ARR);
		  pos1 = rand()/(RAND_MAX/n+1);
		  v		= Z_ARR[pos1]*inj_times;
		  ierr	= VecSetValues(Z,1,&pos1,&v,INSERT_VALUES);CHKERRQ(ierr);
		  VecDestroy(&Z_SEQ);
		  VecScatterDestroy(&ctx);
		  VecAssemblyBegin(Z);
		  VecAssemblyEnd(Z);
		  PetscPrintf(MPI_COMM_WORLD,"Inject an error in Z of position-%d at the end of iteration-%d\n", pos1,i);
		  flag9	= PETSC_FALSE;
	  }
	  
	  /* Inject two errors */
	  if((i==inj_itr)&&(flag10)&&(error_type==10))
	  {
		  srand (time(NULL));
		  VecScatter	ctx;
		  Vec			Z_SEQ;
		  PetscScalar	*Z_ARR;
		  VecScatterCreateToAll(Z,&ctx,&Z_SEQ);
		  VecScatterBegin(ctx,Z,Z_SEQ,INSERT_VALUES,SCATTER_FORWARD);
		  VecScatterEnd(ctx,Z,Z_SEQ,INSERT_VALUES,SCATTER_FORWARD);
		  VecGetArray(Z_SEQ,&Z_ARR);
		  pos1 = rand()/(RAND_MAX/n+1);
		  v		= Z_ARR[pos1]*inj_times;
		  ierr	= VecSetValues(Z,1,&pos1,&v,INSERT_VALUES);CHKERRQ(ierr);
		  pos2 = rand()/(RAND_MAX/n+1);
		  v		= Z_ARR[pos2]*inj_times;
		  ierr	= VecSetValues(Z,1,&pos2,&v,INSERT_VALUES);CHKERRQ(ierr);
		  VecDestroy(&Z_SEQ);
		  VecScatterDestroy(&ctx);
		  VecAssemblyBegin(Z);
		  VecAssemblyEnd(Z);
		  PetscPrintf(MPI_COMM_WORLD,"Inject two errors in Z of position-%d and position-%d at the end of iteration-%d\n", pos1,pos2,i);
		  flag10	= PETSC_FALSE;
		}
	/* Dingwen */
	
  } while (i<ksp->max_it);
  clock_gettime(CLOCK_REALTIME, &end);
  local_diff = 1000000000L*(end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec);
  MPI_Reduce(&local_diff, &global_diff, 1, MPI_LONG_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  PetscPrintf(MPI_COMM_WORLD,"Average time of each MVM = %lf nanoseconds\n",(double)(time_MVM)/num_MVM);
  PetscPrintf(MPI_COMM_WORLD,"Elapsed time of main loop = %lf nanoseconds\n", (double)(global_diff)/size);
  PetscPrintf(MPI_COMM_WORLD,"Number of iterations without rollback = %d\n", i+1);	  
  }

  if (solver_type==3)
  {
	    cg            = (KSP_CG*)ksp->data;
  eigs          = ksp->calc_sings;
  stored_max_it = ksp->max_it;
  X             = ksp->vec_sol;
  B             = ksp->vec_rhs;
  R             = ksp->work[0];
  Z             = ksp->work[1];
  P             = ksp->work[2];
  /* Dingwen */
  CKPX			= ksp->work[3];
  CKPP			= ksp->work[4];
  CKSAmat1		= ksp->work[5];
  CKSAmat2		= ksp->work[6];
  CKSAmat3		= ksp->work[7];
  C1			= ksp->work[8];
  C2			= ksp->work[9];
  C3			= ksp->work[10];
  /* Dingwen */
  
  if (cg->singlereduction) {
    S = ksp->work[11];
    W = ksp->work[12];
  } else {
    S = 0;                      /* unused */
    W = Z;
  }
    
  if (eigs) {e = cg->e; d = cg->d; e[0] = 0.0; }
  ierr = PCGetOperators(ksp->pc,&Amat,&Pmat);CHKERRQ(ierr);
  
  ksp->its = 0;
  if (!ksp->guess_zero) {
    ierr = KSP_MatMult(ksp,Amat,X,R);CHKERRQ(ierr);            /*     r <- b - Ax     */
    ierr = VecAYPX(R,-1.0,B);CHKERRQ(ierr);
  } else {
    ierr = VecCopy(B,R);CHKERRQ(ierr);                         /*     r <- b (x is 0) */
  }
	
  switch (ksp->normtype) {
  case KSP_NORM_PRECONDITIONED:
    ierr = KSP_PCApply(ksp,R,Z);CHKERRQ(ierr);                   /*     z <- Br         */
    ierr = VecNorm(Z,NORM_2,&dp);CHKERRQ(ierr);                /*    dp <- z'*z = e'*A'*B'*B*A'*e'     */
    break;
  case KSP_NORM_UNPRECONDITIONED:
    ierr = VecNorm(R,NORM_2,&dp);CHKERRQ(ierr);                /*    dp <- r'*r = e'*A'*A*e            */
    break;
  case KSP_NORM_NATURAL:
    ierr = KSP_PCApply(ksp,R,Z);CHKERRQ(ierr);                   /*     z <- Br         */
    if (cg->singlereduction) {
      ierr = KSP_MatMult(ksp,Amat,Z,S);CHKERRQ(ierr);
      ierr = VecXDot(Z,S,&delta);CHKERRQ(ierr);
	  /* Dingwen */
	  ierr = VecXDot(C1,S,&CKSS1);CHKERRQ(ierr);						/* Compute the initial checksum1(S) */
	  ierr = VecXDot(C2,S,&CKSS2);CHKERRQ(ierr);						/* Compute the initial checksum2(S) */
	  ierr = VecXDot(C3,S,&CKSS3);CHKERRQ(ierr);						/* Compute the initial checksum3(S) */
	  /* Dingwen */
	}
    ierr = VecXDot(Z,R,&beta);CHKERRQ(ierr);                     /*  beta <- z'*r       */
    KSPCheckDot(ksp,beta);
    dp = PetscSqrtReal(PetscAbsScalar(beta));                           /*    dp <- r'*z = r'*B*r = e'*A'*B*A*e */
    break;
  case KSP_NORM_NONE:
    dp = 0.0;
    break;
  default: SETERRQ1(PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"%s",KSPNormTypes[ksp->normtype]);
  }
  
  ierr       = KSPLogResidualHistory(ksp,dp);CHKERRQ(ierr);
  ierr       = KSPMonitor(ksp,0,dp);CHKERRQ(ierr);
  ksp->rnorm = dp;

  ierr = (*ksp->converged)(ksp,0,dp,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);      /* test for convergence */
  if (ksp->reason) PetscFunctionReturn(0);

  if (ksp->normtype != KSP_NORM_PRECONDITIONED && (ksp->normtype != KSP_NORM_NATURAL)) {
    ierr = KSP_PCApply(ksp,R,Z);CHKERRQ(ierr);                   /*     z <- Br         */
  }
  if (ksp->normtype != KSP_NORM_NATURAL) {
    if (cg->singlereduction) {
      ierr = KSP_MatMult(ksp,Amat,Z,S);CHKERRQ(ierr);
      ierr = VecXDot(Z,S,&delta);CHKERRQ(ierr);
    }
    ierr = VecXDot(Z,R,&beta);CHKERRQ(ierr);         /*  beta <- z'*r       */
    KSPCheckDot(ksp,beta);
  }

  /* Dingwen */
  /* checksum coefficients initialization */
  PetscInt n;
  PetscInt *index;
  PetscScalar *v1,*v2,*v3;
  ierr = VecGetSize(B,&n);
  v1 	= (PetscScalar *)malloc(n*sizeof(PetscScalar));
  v2 	= (PetscScalar *)malloc(n*sizeof(PetscScalar));
  v3 	= (PetscScalar *)malloc(n*sizeof(PetscScalar));
  index	= (PetscInt *)malloc(n*sizeof(PetscInt));
  for (i=0; i<n; i++)
  {
	  index[i] = i;
	  v1[i] = 1.0;
	  v2[i] = i+1.0;
	  v3[i] = 1/(i+1.0);
  }
  ierr	= VecSetValues(C1,n,index,v1,INSERT_VALUES);CHKERRQ(ierr);	
  ierr 	= VecSetValues(C2,n,index,v2,INSERT_VALUES);CHKERRQ(ierr);
  ierr	= VecSetValues(C3,n,index,v3,INSERT_VALUES);CHKERRQ(ierr);	
  d1 = 1.0;
  d2 = 2.0;
  d3 = 3.0;
  ierr = KSP_MatMultTranspose(ksp,Amat,C1,CKSAmat1);CHKERRQ(ierr);
  ierr = VecAXPY(CKSAmat1,-d1,C1);CHKERRQ(ierr);
  ierr = VecAXPY(CKSAmat1,-d2,C2);CHKERRQ(ierr); 
  ierr = VecAXPY(CKSAmat1,-d3,C3);CHKERRQ(ierr);					/* Compute the initial checksum1(A) */ 
  ierr = KSP_MatMultTranspose(ksp,Amat,C2,CKSAmat2);CHKERRQ(ierr);
  ierr = VecAXPY(CKSAmat2,-d2,C1);CHKERRQ(ierr);
  ierr = VecAXPY(CKSAmat2,-d3,C2);CHKERRQ(ierr);
  ierr = VecAXPY(CKSAmat2,-d1,C3);CHKERRQ(ierr);					/* Compute the initial checksum2(A) */ 
  ierr = KSP_MatMultTranspose(ksp,Amat,C3,CKSAmat3);CHKERRQ(ierr);
  ierr = VecAXPY(CKSAmat3,-d3,C1);CHKERRQ(ierr);
  ierr = VecAXPY(CKSAmat3,-d1,C2);CHKERRQ(ierr);
  ierr = VecAXPY(CKSAmat3,-d2,C3);CHKERRQ(ierr);					/* Compute the initial checksum3(A) */
  /* Checksum Initialization */
  ierr = VecXDot(C1,X,&CKSX1);CHKERRQ(ierr);						/* Compute the initial checksum1(X) */ 
  ierr = VecXDot(C1,R,&CKSR1);CHKERRQ(ierr);						/* Compute the initial checksum1(R) */
  ierr = VecXDot(C1,Z,&CKSZ1);CHKERRQ(ierr);						/* Compute the initial checksum1(Z) */
  ierr = VecXDot(C2,X,&CKSX2);CHKERRQ(ierr);						/* Compute the initial checksum2(X) */ 
  ierr = VecXDot(C2,R,&CKSR2);CHKERRQ(ierr);						/* Compute the initial checksum2(R) */
  ierr = VecXDot(C2,Z,&CKSZ2);CHKERRQ(ierr);						/* Compute the initial checksum2(Z) */
  ierr = VecXDot(C3,X,&CKSX3);CHKERRQ(ierr);						/* Compute the initial checksum3(X) */ 
  ierr = VecXDot(C3,R,&CKSR3);CHKERRQ(ierr);						/* Compute the initial checksum3(R) */
  ierr = VecXDot(C3,Z,&CKSZ3);CHKERRQ(ierr);						/* Compute the initial checksum3(Z) */

  struct timespec start, end;
  long long int local_diff, global_diff;
  clock_gettime(CLOCK_REALTIME, &start);
  
  i = 0;
  do {
	  /* Dingwen */
	  if ((i>0) && (i%itv_d == 0))
	  {
		  PetscScalar	sumX1,sumR1;
		  ierr = VecXDot(C1,X,&sumX1);CHKERRQ(ierr);
		  ierr = VecXDot(C1,R,&sumR1);CHKERRQ(ierr);
		  if ((PetscAbsScalar(sumX1-CKSX1)/(n*n) > 1.0e-6) || (PetscAbsScalar(sumR1-CKSR1)/(n*n) > 1.0e-6))
		  {
			  /* Rollback and Recovery */
			  PetscPrintf (MPI_COMM_WORLD,"Recovery start...\n");
			  PetscPrintf (MPI_COMM_WORLD,"Rollback from iteration-%d to iteration-%d\n",i,CKPi);
			  betaold = CKPbetaold;									/* Recovery scalar betaold by checkpoint*/
			  i = CKPi;													/* Recovery integer i by checkpoint */
			  ierr = VecCopy(CKPP,P);CHKERRQ(ierr);						/* Recovery vector P from checkpoint */
			  ierr = VecXDot(C1,P,&CKSP1);CHKERRQ(ierr);				/* Recovery checksum1(P) by P */	
			  ierr = VecXDot(C2,P,&CKSP2);CHKERRQ(ierr);				/* Recovery checksum2(P) by P */			  
			  ierr = VecXDot(C3,P,&CKSP3);CHKERRQ(ierr);				/* Recovery checksum3(P) by P */
			  ierr = KSP_MatMult(ksp,Amat,P,W);CHKERRQ(ierr);			/* Recovery vector W by P */
			  ierr = VecXDot(P,W,&dpi);CHKERRQ(ierr);					/* Recovery scalar dpi by P and W */
			  ierr = VecCopy(CKPX,X);CHKERRQ(ierr);						/* Recovery vector X from checkpoint */
			  ierr = VecXDot(C1,X,&CKSX1);CHKERRQ(ierr);				/* Recovery checksum1(X) by X */
			  ierr = VecXDot(C2,X,&CKSX2);CHKERRQ(ierr);				/* Recovery checksum2(X) by X */ 			  
			  ierr = VecXDot(C3,X,&CKSX3);CHKERRQ(ierr);				/* Recovery checksum3(X) by X */ 			  
			  ierr = KSP_MatMult(ksp,Amat,X,R);CHKERRQ(ierr);			/* Recovery vector R by X */
			  ierr = VecAYPX(R,-1.0,B);CHKERRQ(ierr);
			  ierr = VecXDot(C1,R,&CKSR1);CHKERRQ(ierr);				/* Recovery checksum1(R) by R */
			  ierr = VecXDot(C2,R,&CKSR2);CHKERRQ(ierr);				/* Recovery checksum2(R) by R */
			  ierr = VecXDot(C3,R,&CKSR3);CHKERRQ(ierr);				/* Recovery checksum3(R) by R */
			  ierr = KSP_PCApply(ksp,R,Z);CHKERRQ(ierr);				/* Recovery vector Z by R */
			  ierr = VecXDot(C1,Z,&CKSZ1);CHKERRQ(ierr);				/* Recovery checksum1(Z) by Z */
			  ierr = VecXDot(C2,Z,&CKSZ2);CHKERRQ(ierr);				/* Recovery checksum2(Z) by Z */
			  ierr = VecXDot(C3,Z,&CKSZ3);CHKERRQ(ierr);				/* Recovery checksum3(Z) by Z */
			  ierr = VecXDot(Z,R,&beta);CHKERRQ(ierr);					/* Recovery scalar beta by Z and R */
			  PetscPrintf (MPI_COMM_WORLD,"Recovery end.\n");
		}
		else if (i%(itv_c*itv_d) == 0)
		{
			ierr = VecCopy(X,CKPX);CHKERRQ(ierr);
			ierr = VecCopy(P,CKPP);CHKERRQ(ierr);
			CKPbetaold = betaold;
			CKPi = i;
		}
	}
	  //ksp->its = i+1;
	  ksp->its++;
	  if (beta == 0.0) {
      ksp->reason = KSP_CONVERGED_ATOL;
      ierr        = PetscInfo(ksp,"converged due to beta = 0\n");CHKERRQ(ierr);
      break;
#if !defined(PETSC_USE_COMPLEX)
    } else if ((i > 0) && (beta*betaold < 0.0)) {
      ksp->reason = KSP_DIVERGED_INDEFINITE_PC;
      ierr        = PetscInfo(ksp,"diverging due to indefinite preconditioner\n");CHKERRQ(ierr);
      break;
#endif
    }
    if (!i) {
      ierr = VecCopy(Z,P);CHKERRQ(ierr);         /*     p <- z          */
      b    = 0.0;
	  /* Dingwen */
	  ierr = VecXDot(C1,P, &CKSP1);CHKERRQ(ierr);  				/* Compute the initial checksum1(P) */
	  ierr = VecXDot(C2,P, &CKSP2);CHKERRQ(ierr);  				/* Compute the initial checksum2(P) */
	  ierr = VecXDot(C3,P, &CKSP3);CHKERRQ(ierr);  				/* Compute the initial checksum3(P) */
	  /* Dingwen */
    } else {
      b = beta/betaold;
      if (eigs) {
        if (ksp->max_it != stored_max_it) SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"Can not change maxit AND calculate eigenvalues");
        e[i] = PetscSqrtReal(PetscAbsScalar(b))/a;
      }
      ierr = VecAYPX(P,b,Z);CHKERRQ(ierr);    /*     p <- z + b* p   */	  
	  /* Dingwen */
	  CKSP1 = CKSZ1 + b*CKSP1;										/* Update checksum1(P) = checksum1(Z) + b*checksum1(P); */
	  CKSP2 = CKSZ2 + b*CKSP2;										/* Update checksum2(P) = checksum2(Z) + b*checksum2(P); */
	  CKSP3 = CKSZ3 + b*CKSP3;										/* Update checksum3(P) = checksum3(Z) + b*checksum3(P); */
	  /* Dingwen */
    }
    dpiold = dpi;
    if (!cg->singlereduction || !i) {	  
	  ierr = KSP_MatMult(ksp,Amat,P,W);CHKERRQ(ierr);          /*     w <- Ap         */	/* MVM */
      ierr = VecXDot(P,W,&dpi);CHKERRQ(ierr);                  /*     dpi <- p'w     */	  
	  
	  /* Dingwen */
	  ierr = VecXDot(CKSAmat1, P, &CKSW1);CHKERRQ(ierr);
	  CKSW1 = CKSW1 + d1*CKSP1 + d2*CKSP2 + d3*CKSP3;									/* Update checksum1(W) = checksum1(A)P + d1*checksum1(P) + d2*checksum2(P) + d3*checksum3(P); */
	  ierr = VecXDot(CKSAmat2, P, &CKSW2);CHKERRQ(ierr);
	  CKSW2 = CKSW2 + d2*CKSP1 + d3*CKSP2 + d1*CKSP3;									/* Update checksum2(W) = checksum2(A)P + d2*checksum1(P) + d3*checksum2(P) + d1*checksum3(P); */
	  ierr = VecXDot(CKSAmat3, P, &CKSW3);CHKERRQ(ierr);
	  CKSW3 = CKSW3 + d3*CKSP1 + d1*CKSP2 + d2*CKSP3;									/* Update checksum3(W) = checksum3(A)P + d3*checksum1(P) + d1*checksum2(P) + d2*checksum3(P); */

	  /* Inject an error */
	  if((i==inj_itr)&&(flag1)&&(error_type==1))
	  {
		  srand (time(NULL));
		  VecScatter	ctx;
		  Vec			W_SEQ;
		  PetscScalar	*W_ARR;
		  VecScatterCreateToAll(W,&ctx,&W_SEQ);
		  VecScatterBegin(ctx,W,W_SEQ,INSERT_VALUES,SCATTER_FORWARD);
		  VecScatterEnd(ctx,W,W_SEQ,INSERT_VALUES,SCATTER_FORWARD);
		  VecGetArray(W_SEQ,&W_ARR);
		  pos1 = rand()/(RAND_MAX/n+1);
		  v		= W_ARR[pos1]*inj_times;
		  ierr	= VecSetValues(W,1,&pos1,&v,INSERT_VALUES);CHKERRQ(ierr);
		  VecDestroy(&W_SEQ);
		  VecScatterDestroy(&ctx);
		  VecAssemblyBegin(W);
		  VecAssemblyEnd(W);
		  PetscPrintf(MPI_COMM_WORLD,"Inject an error in W of position-%d after MVM at iteration-%d\n", pos1,i);
		  flag1	= PETSC_FALSE;
	  }
	  
	  /* Inject two errors */
	  if((i==inj_itr)&&(flag2)&&(error_type==2))
	  {
		  srand (time(NULL));
		  VecScatter	ctx;
		  Vec			W_SEQ;
		  PetscScalar	*W_ARR;
		  VecScatterCreateToAll(W,&ctx,&W_SEQ);
		  VecScatterBegin(ctx,W,W_SEQ,INSERT_VALUES,SCATTER_FORWARD);
		  VecScatterEnd(ctx,W,W_SEQ,INSERT_VALUES,SCATTER_FORWARD);
		  VecGetArray(W_SEQ,&W_ARR);
		  pos1 = rand()/(RAND_MAX/n+1);
		  v		= W_ARR[pos1]*inj_times;
		  ierr	= VecSetValues(W,1,&pos1,&v,INSERT_VALUES);CHKERRQ(ierr);
		  pos2 = rand()/(RAND_MAX/n+1);
		  v		= W_ARR[pos2]*inj_times;
		  ierr	= VecSetValues(W,1,&pos2,&v,INSERT_VALUES);CHKERRQ(ierr);
		  VecDestroy(&W_SEQ);
		  VecScatterDestroy(&ctx);
		  VecAssemblyBegin(W);
		  VecAssemblyEnd(W);
		  PetscPrintf(MPI_COMM_WORLD,"Inject two errors in W of position-%d and position-%d after MVM at iteration-%d\n", pos1,pos2,i);
		  flag2	= PETSC_FALSE;
		}

	  /* Inner Protection */
	  PetscScalar delta1,delta2,delta3;			  
	  PetscScalar sumW1,sumW2,sumW3;	  
	  ierr = VecXDot(C1,W,&sumW1);CHKERRQ(ierr);
	  ierr = VecXDot(C2,W,&sumW2);CHKERRQ(ierr);
	  ierr = VecXDot(C3,W,&sumW3);CHKERRQ(ierr);
	  delta1 = sumW1 - CKSW1;
	  delta2 = sumW2 - CKSW2;
	  delta3 = sumW3 - CKSW3;
	  if (PetscAbsScalar(delta1)/(n*n) > 1.0e-6)
	  {
		  PetscScalar sumP1;
		  ierr = VecXDot(C1,P,&sumP1);CHKERRQ(ierr);
		  if (PetscAbsScalar(CKSP1-sumP1)/(n*n) > 1.0e-6)
		  {
			  /* Rollback and Recovery */
			  PetscPrintf(MPI_COMM_WORLD,"Errors occur before MVM\n");
			  PetscPrintf(MPI_COMM_WORLD,"Recovery start...\n");
			  PetscPrintf(MPI_COMM_WORLD,"Rollback from iteration-%d to iteration-%d\n",i,CKPi);
			  betaold = CKPbetaold;										/* Recovery scalar betaold by checkpoint*/
			  i = CKPi;													/* Recovery integer i by checkpoint */
			  ierr = VecCopy(CKPP,P);CHKERRQ(ierr);						/* Recovery vector P from checkpoint */
			  ierr = VecXDot(C1,P,&CKSP1);CHKERRQ(ierr);				/* Recovery checksum1(P) by P */	
			  ierr = VecXDot(C2,P,&CKSP2);CHKERRQ(ierr);				/* Recovery checksum2(P) by P */			  
			  ierr = VecXDot(C3,P,&CKSP3);CHKERRQ(ierr);				/* Recovery checksum3(P) by P */
			  ierr = KSP_MatMult(ksp,Amat,P,W);CHKERRQ(ierr);			/* Recovery vector W by P */
			  ierr = VecXDot(P,W,&dpi);CHKERRQ(ierr);					/* Recovery scalar dpi by P and W */
			  ierr = VecCopy(CKPX,X);CHKERRQ(ierr);						/* Recovery vector X from checkpoint */
			  ierr = VecXDot(C1,X,&CKSX1);CHKERRQ(ierr);				/* Recovery checksum1(X) by X */
			  ierr = VecXDot(C2,X,&CKSX2);CHKERRQ(ierr);				/* Recovery checksum2(X) by X */ 			  
			  ierr = VecXDot(C3,X,&CKSX3);CHKERRQ(ierr);				/* Recovery checksum3(X) by X */ 			  
			  ierr = KSP_MatMult(ksp,Amat,X,R);CHKERRQ(ierr);			/* Recovery vector R by X */
			  ierr = VecAYPX(R,-1.0,B);CHKERRQ(ierr);
			  ierr = VecXDot(C1,R,&CKSR1);CHKERRQ(ierr);				/* Recovery checksum1(R) by R */
			  ierr = VecXDot(C2,R,&CKSR2);CHKERRQ(ierr);				/* Recovery checksum2(R) by R */
			  ierr = VecXDot(C3,R,&CKSR3);CHKERRQ(ierr);				/* Recovery checksum3(R) by R */
			  ierr = KSP_PCApply(ksp,R,Z);CHKERRQ(ierr);				/* Recovery vector Z by R */
			  ierr = VecXDot(C1,Z,&CKSZ1);CHKERRQ(ierr);				/* Recovery checksum1(Z) by Z */
			  ierr = VecXDot(C2,Z,&CKSZ2);CHKERRQ(ierr);				/* Recovery checksum2(Z) by Z */
			  ierr = VecXDot(C3,Z,&CKSZ3);CHKERRQ(ierr);				/* Recovery checksum3(Z) by Z */
			  ierr = VecXDot(Z,R,&beta);CHKERRQ(ierr);					/* Recovery scalar beta by Z and R */
			  PetscPrintf(MPI_COMM_WORLD,"Recovery end.\n");
			  
			  /* Recover the calculations from iteration begining to checking */
			  b = beta/betaold;
			  ierr = VecAYPX(P,b,Z);CHKERRQ(ierr);    					/*     p <- z + b* p   */	  
			  CKSP1 = CKSZ1 + b*CKSP1;									/* Update checksum1(P) = checksum1(Z) + b*checksum1(P); */
			  CKSP2 = CKSZ2 + b*CKSP2;									/* Update checksum2(P) = checksum2(Z) + b*checksum2(P); */
			  CKSP3 = CKSZ3 + b*CKSP3;									/* Update checksum3(P) = checksum3(Z) + b*checksum3(P); */
			  ierr = KSP_MatMult(ksp,Amat,P,W);CHKERRQ(ierr);			/*     w <- Ap         */	/* MVM */
			  ierr = VecXDot(P,W,&dpi);CHKERRQ(ierr);					/*     dpi <- p'w     */	  
			  ierr = VecXDot(CKSAmat1, P, &CKSW1);CHKERRQ(ierr);
			  CKSW1 = CKSW1 + d1*CKSP1 + d2*CKSP2 + d3*CKSP3;			/* Update checksum1(W) = checksum1(A)P + d1*checksum1(P) + d2*checksum2(P) + d3*checksum3(P); */
			  ierr = VecXDot(CKSAmat2, P, &CKSW2);CHKERRQ(ierr);
			  CKSW2 = CKSW2 + d2*CKSP1 + d3*CKSP2 + d1*CKSP3;			/* Update checksum2(W) = checksum2(A)P + d2*checksum1(P) + d3*checksum2(P) + d1*checksum3(P); */
			  ierr = VecXDot(CKSAmat3, P, &CKSW3);CHKERRQ(ierr);
			  CKSW3 = CKSW3 + d3*CKSP1 + d1*CKSP2 + d2*CKSP3;			/* Update checksum3(W) = checksum3(A)P + d3*checksum1(P) + d1*checksum2(P) + d2*checksum3(P); */
		  }
		  else{
			  if (PetscAbsScalar(1.0-(delta2*delta3)/(delta1*delta1)) > 1.0e-6)
			  {
			  /* Rollback and Recovery */
			  PetscPrintf(MPI_COMM_WORLD,"Multiple errors of output vector\n");
			  PetscPrintf(MPI_COMM_WORLD,"Recovery start...\n");
			  PetscPrintf(MPI_COMM_WORLD,"Rollback from iteration-%d to iteration-%d\n",i,CKPi);
			  betaold = CKPbetaold;										/* Recovery scalar betaold by checkpoint*/
			  i = CKPi;													/* Recovery integer i by checkpoint */
			  ierr = VecCopy(CKPP,P);CHKERRQ(ierr);						/* Recovery vector P from checkpoint */
			  ierr = VecXDot(C1,P,&CKSP1);CHKERRQ(ierr);				/* Recovery checksum1(P) by P */	
			  ierr = VecXDot(C2,P,&CKSP2);CHKERRQ(ierr);				/* Recovery checksum2(P) by P */			  
			  ierr = VecXDot(C3,P,&CKSP3);CHKERRQ(ierr);				/* Recovery checksum3(P) by P */
			  ierr = KSP_MatMult(ksp,Amat,P,W);CHKERRQ(ierr);			/* Recovery vector W by P */
			  ierr = VecXDot(P,W,&dpi);CHKERRQ(ierr);					/* Recovery scalar dpi by P and W */
			  ierr = VecCopy(CKPX,X);CHKERRQ(ierr);						/* Recovery vector X from checkpoint */
			  ierr = VecXDot(C1,X,&CKSX1);CHKERRQ(ierr);				/* Recovery checksum1(X) by X */
			  ierr = VecXDot(C2,X,&CKSX2);CHKERRQ(ierr);				/* Recovery checksum2(X) by X */ 			  
			  ierr = VecXDot(C3,X,&CKSX3);CHKERRQ(ierr);				/* Recovery checksum3(X) by X */ 			  
			  ierr = KSP_MatMult(ksp,Amat,X,R);CHKERRQ(ierr);			/* Recovery vector R by X */
			  ierr = VecAYPX(R,-1.0,B);CHKERRQ(ierr);
			  ierr = VecXDot(C1,R,&CKSR1);CHKERRQ(ierr);				/* Recovery checksum1(R) by R */
			  ierr = VecXDot(C2,R,&CKSR2);CHKERRQ(ierr);				/* Recovery checksum2(R) by R */
			  ierr = VecXDot(C3,R,&CKSR3);CHKERRQ(ierr);				/* Recovery checksum3(R) by R */
			  ierr = KSP_PCApply(ksp,R,Z);CHKERRQ(ierr);				/* Recovery vector Z by R */
			  ierr = VecXDot(C1,Z,&CKSZ1);CHKERRQ(ierr);				/* Recovery checksum1(Z) by Z */
			  ierr = VecXDot(C2,Z,&CKSZ2);CHKERRQ(ierr);				/* Recovery checksum2(Z) by Z */
			  ierr = VecXDot(C3,Z,&CKSZ3);CHKERRQ(ierr);				/* Recovery checksum3(Z) by Z */
			  ierr = VecXDot(Z,R,&beta);CHKERRQ(ierr);					/* Recovery scalar beta by Z and R */
			  PetscPrintf(MPI_COMM_WORLD,"Recovery end.\n");
			  
			  /* Recover the calculations from iteration begining to checking */
			  b = beta/betaold;
			  ierr = VecAYPX(P,b,Z);CHKERRQ(ierr);    					/*     p <- z + b* p   */	  
			  CKSP1 = CKSZ1 + b*CKSP1;									/* Update checksum1(P) = checksum1(Z) + b*checksum1(P); */
			  CKSP2 = CKSZ2 + b*CKSP2;									/* Update checksum2(P) = checksum2(Z) + b*checksum2(P); */
			  CKSP3 = CKSZ3 + b*CKSP3;									/* Update checksum3(P) = checksum3(Z) + b*checksum3(P); */
			  ierr = KSP_MatMult(ksp,Amat,P,W);CHKERRQ(ierr);			/*     w <- Ap         */	/* MVM */
			  ierr = VecXDot(P,W,&dpi);CHKERRQ(ierr);					/*     dpi <- p'w     */	  
			  ierr = VecXDot(CKSAmat1, P, &CKSW1);CHKERRQ(ierr);
			  CKSW1 = CKSW1 + d1*CKSP1 + d2*CKSP2 + d3*CKSP3;			/* Update checksum1(W) = checksum1(A)P + d1*checksum1(P) + d2*checksum2(P) + d3*checksum3(P); */
			  ierr = VecXDot(CKSAmat2, P, &CKSW2);CHKERRQ(ierr);
			  CKSW2 = CKSW2 + d2*CKSP1 + d3*CKSP2 + d1*CKSP3;			/* Update checksum2(W) = checksum2(A)P + d2*checksum1(P) + d3*checksum2(P) + d1*checksum3(P); */
			  ierr = VecXDot(CKSAmat3, P, &CKSW3);CHKERRQ(ierr);
			  CKSW3 = CKSW3 + d3*CKSP1 + d1*CKSP2 + d2*CKSP3;			/* Update checksum3(W) = checksum3(A)P + d3*checksum1(P) + d1*checksum2(P) + d2*checksum3(P); */			  
			  }
			  else{
				  if (rank == 0) printf ("Locate and correct right away\n");
				  VecScatter	ctx;
				  Vec			W_SEQ;
				  PetscScalar	*W_ARR;
				  VecScatterCreateToAll(W,&ctx,&W_SEQ);
				  VecScatterBegin(ctx,W,W_SEQ,INSERT_VALUES,SCATTER_FORWARD);
				  VecScatterEnd(ctx,W,W_SEQ,INSERT_VALUES,SCATTER_FORWARD);
				  VecGetArray(W_SEQ,&W_ARR);
				  pos1	= rint(delta2/delta1) - 1;
				  v		= W_ARR[pos1];
				  v		= v - delta1;
				  ierr	= VecSetValues(W,1,&pos1,&v,INSERT_VALUES);CHKERRQ(ierr);
				  PetscPrintf(MPI_COMM_WORLD,"Correct an error in output vector of MVM at iteration-%d\n", i);
				  VecDestroy(&W_SEQ);
				  VecScatterDestroy(&ctx);
			  }
		  }
		}
		/* Dingwen */
		
    } else {
      ierr = VecAYPX(W,beta/betaold,S);CHKERRQ(ierr);                  /*     w <- Ap         */
      dpi  = delta - beta*beta*dpiold/(betaold*betaold);             /*     dpi <- p'w     */
	  /* Dingwen */
	  CKSW1 = beta/betaold*CKSW1 + CKSS1;							/* Update checksum1(W) = checksum1(S) + beta/betaold*checksum1(W); */
	  CKSW2 = beta/betaold*CKSW2 + CKSS2;							/* Update checksum2(W) = checksum2(S) + beta/betaold*checksum2(W); */
	  CKSW3 = beta/betaold*CKSW3 + CKSS3;							/* Update checksum3(W) = checksum3(S) + beta/betaold*checksum3(W); */
	  /* Dingwen */
	}
    betaold = beta;
    KSPCheckDot(ksp,beta);

    if ((dpi == 0.0) || ((i > 0) && (PetscRealPart(dpi*dpiold) <= 0.0))) {
      ksp->reason = KSP_DIVERGED_INDEFINITE_MAT;
      ierr        = PetscInfo(ksp,"diverging due to indefinite or negative definite matrix\n");CHKERRQ(ierr);
      break;
    }
    a = beta/dpi;                                 /*     a = beta/p'w   */
    if (eigs) d[i] = PetscSqrtReal(PetscAbsScalar(b))*e[i] + 1.0/a;
    ierr = VecAXPY(X,a,P);CHKERRQ(ierr);          /*     x <- x + ap     */
	/* Dingwen */
	CKSX1 = CKSX1 + a*CKSP1;									/* Update checksum1(X) = checksum1(X) + a*checksum1(P); */
	CKSX2 = CKSX2 + a*CKSP2;									/* Update checksum2(X) = checksum2(X) + a*checksum2(P); */
	CKSX3 = CKSX3 + a*CKSP3;									/* Update checksum3(X) = checksum3(X) + a*checksum3(P); */
	/* Dingwen */
    
	ierr = VecAXPY(R,-a,W);CHKERRQ(ierr);                      /*     r <- r - aw    */

	/* Dingwen */
	CKSR1 = CKSR1 - a*CKSW1;									/* Update checksum1(R) = checksum1(R) - a*checksum1(W); */
	CKSR2 = CKSR2 - a*CKSW2;									/* Update checksum2(R) = checksum2(R) - a*checksum2(W); */
	CKSR3 = CKSR3 - a*CKSW3;									/* Update checksum3(R) = checksum3(R) - a*checksum3(W); */
	/* Dingwen */
	
	if (ksp->normtype == KSP_NORM_PRECONDITIONED && ksp->chknorm < i+2) {      
	  ierr = KSP_PCApply(ksp,R,Z);CHKERRQ(ierr);               /*     z <- Br         */
	  
	  /* Dingwen */
	  ierr = VecXDot(C1,Z, &CKSZ1);CHKERRQ(ierr);				/* Update checksum1(Z) */
	  ierr = VecXDot(C2,Z, &CKSZ2);CHKERRQ(ierr);				/* Update checksum2(Z) */
	  ierr = VecXDot(C3,Z, &CKSZ3);CHKERRQ(ierr);				/* Update checksum3(Z) */
	  /* Dingwen */
	  
	  if (cg->singlereduction) {
        ierr = KSP_MatMult(ksp,Amat,Z,S);CHKERRQ(ierr);				/* MVM */
		/* Dingwen */
		ierr = VecXDot(CKSAmat1, Z, &CKSS1);CHKERRQ(ierr);
		CKSS1 = CKSS1 + d1*CKSZ1 + d2*CKSZ2 + d3*CKSZ3;						/* Update checksum1(S) = checksum1(A)Z + d1*chekcsum1(Z) + d2*checksum2(Z) + d3*checksum3(Z); */
		ierr = VecXDot(CKSAmat2, Z, &CKSS2);CHKERRQ(ierr);
		CKSS2 = CKSS2 + d2*CKSZ1 + d3*CKSZ2 + d1*CKSZ3;						/* Update checksum2(S) = checksum2(A)Z + d2*chekcsum1(Z) + d3*checksum2(Z) + d1*checksum3(Z); */
		ierr = VecXDot(CKSAmat3, Z, &CKSS3);CHKERRQ(ierr);
		CKSS3 = CKSS3 + d3*CKSZ1 + d1*CKSZ2 + d2*CKSZ3;						/* Update checksum3(S) = checksum3(A)Z + d3*chekcsum1(Z) + d1*checksum2(Z) + d2*checksum3(Z); */

		/* Dingwen */
      }
      ierr = VecNorm(Z,NORM_2,&dp);CHKERRQ(ierr);              /*    dp <- z'*z       */
    } else if (ksp->normtype == KSP_NORM_UNPRECONDITIONED && ksp->chknorm < i+2) {
      ierr = VecNorm(R,NORM_2,&dp);CHKERRQ(ierr);              /*    dp <- r'*r       */
    } else if (ksp->normtype == KSP_NORM_NATURAL) {
      ierr = KSP_PCApply(ksp,R,Z);CHKERRQ(ierr);               /*     z <- Br         */
 	  
	  /* Dingwen */
	  ierr = VecXDot(C1,Z, &CKSZ1);CHKERRQ(ierr);				/* Update checksum1(Z) */
	  ierr = VecXDot(C2,Z, &CKSZ2);CHKERRQ(ierr);				/* Update checksum2(Z) */
	  ierr = VecXDot(C3,Z, &CKSZ3);CHKERRQ(ierr);				/* Update checksum3(Z) */	  
	  /* Dingwen */
	  
	  if (cg->singlereduction) {
        PetscScalar tmp[2];
        Vec         vecs[2];
        vecs[0] = S; vecs[1] = R;
        ierr    = KSP_MatMult(ksp,Amat,Z,S);CHKERRQ(ierr);
        ierr  = VecMDot(Z,2,vecs,tmp);CHKERRQ(ierr);
        delta = tmp[0]; beta = tmp[1];
      } else {
        ierr = VecXDot(Z,R,&beta);CHKERRQ(ierr);     /*  beta <- r'*z       */
      }
      KSPCheckDot(ksp,beta);
      dp = PetscSqrtReal(PetscAbsScalar(beta));
    } else {
      dp = 0.0;
    }
	  
    ksp->rnorm = dp;
    CHKERRQ(ierr);KSPLogResidualHistory(ksp,dp);CHKERRQ(ierr);
    ierr = KSPMonitor(ksp,i+1,dp);CHKERRQ(ierr);
    ierr = (*ksp->converged)(ksp,i+1,dp,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
    if (ksp->reason) break;

    if ((ksp->normtype != KSP_NORM_PRECONDITIONED && (ksp->normtype != KSP_NORM_NATURAL)) || (ksp->chknorm >= i+2)) {
      ierr = KSP_PCApply(ksp,R,Z);CHKERRQ(ierr);                   /*     z <- Br         */
	  
	  /* Dingwen */
	  ierr = VecXDot(C1,Z, &CKSZ1);CHKERRQ(ierr);				/* Update checksum1(Z) */
	  ierr = VecXDot(C2,Z, &CKSZ2);CHKERRQ(ierr);				/* Update checksum2(Z) */ 
	  ierr = VecXDot(C3,Z, &CKSZ3);CHKERRQ(ierr);				/* Update checksum3(Z) */ 
	  /* Dingwen */
      
	  if (cg->singlereduction) {
        ierr = KSP_MatMult(ksp,Amat,Z,S);CHKERRQ(ierr);
      }
    }
		  
    if ((ksp->normtype != KSP_NORM_NATURAL) || (ksp->chknorm >= i+2)) {
      if (cg->singlereduction) {
        PetscScalar tmp[2];
        Vec         vecs[2];
        vecs[0] = S; vecs[1] = R;
        ierr  = VecMDot(Z,2,vecs,tmp);CHKERRQ(ierr);
        delta = tmp[0]; beta = tmp[1];
      } else {
        ierr = VecXDot(Z,R,&beta);CHKERRQ(ierr);        /*  beta <- z'*r       */
      }
      KSPCheckDot(ksp,beta);
    }
	
    i++;
	
	/* Dingwen */
	/* Inject an error in P */
		  if((i==inj_itr)&&(flag3)&&(error_type==3))
	  {
		  srand (time(NULL));
		  VecScatter	ctx;
		  Vec			P_SEQ;
		  PetscScalar	*P_ARR;
		  VecScatterCreateToAll(P,&ctx,&P_SEQ);
		  VecScatterBegin(ctx,P,P_SEQ,INSERT_VALUES,SCATTER_FORWARD);
		  VecScatterEnd(ctx,P,P_SEQ,INSERT_VALUES,SCATTER_FORWARD);
		  VecGetArray(P_SEQ,&P_ARR);
		  pos1 = rand()/(RAND_MAX/n+1);
		  v		= P_ARR[pos1]*inj_times;
		  ierr	= VecSetValues(P,1,&pos1,&v,INSERT_VALUES);CHKERRQ(ierr);
		  VecDestroy(&P_SEQ);
		  VecScatterDestroy(&ctx);
		  VecAssemblyBegin(P);
		  VecAssemblyEnd(P);
		  PetscPrintf(MPI_COMM_WORLD,"Inject an error in P of position-%d at the end of iteration-%d\n", pos1,i);
		  flag3	= PETSC_FALSE;
	  }
	  
	  /* Inject two errors in P */
	  if((i==inj_itr)&&(flag4)&(error_type==4))
	  {
		  srand (time(NULL));
		  VecScatter	ctx;
		  Vec			P_SEQ;
		  PetscScalar	*P_ARR;
		  VecScatterCreateToAll(P,&ctx,&P_SEQ);
		  VecScatterBegin(ctx,P,P_SEQ,INSERT_VALUES,SCATTER_FORWARD);
		  VecScatterEnd(ctx,P,P_SEQ,INSERT_VALUES,SCATTER_FORWARD);
		  VecGetArray(P_SEQ,&P_ARR);
		  pos1 = rand()/(RAND_MAX/n+1);
		  v		= P_ARR[pos1]*inj_times;
		  ierr	= VecSetValues(P,1,&pos1,&v,INSERT_VALUES);CHKERRQ(ierr);
		  pos2 = rand()/(RAND_MAX/n+1);
		  v		= P_ARR[pos2]*inj_times;
		  ierr	= VecSetValues(P,1,&pos2,&v,INSERT_VALUES);CHKERRQ(ierr);
		  VecDestroy(&P_SEQ);
		  VecScatterDestroy(&ctx);
		  VecAssemblyBegin(P);
		  VecAssemblyEnd(P);
		  PetscPrintf(MPI_COMM_WORLD,"Inject two errors in P of position-%d and position-%d at the end of iteration-%d\n", pos1,pos2,i);
		  flag4	= PETSC_FALSE;
		}
		
		/* Inject an error in X*/
	  if((i==inj_itr)&&(flag5)&&(error_type==5))
	  {
		  srand (time(NULL));
		  VecScatter	ctx;
		  Vec			X_SEQ;
		  PetscScalar	*X_ARR;
		  VecScatterCreateToAll(X,&ctx,&X_SEQ);
		  VecScatterBegin(ctx,X,X_SEQ,INSERT_VALUES,SCATTER_FORWARD);
		  VecScatterEnd(ctx,X,X_SEQ,INSERT_VALUES,SCATTER_FORWARD);
		  VecGetArray(X_SEQ,&X_ARR);
		  pos1 = rand()/(RAND_MAX/n+1);
		  v		= X_ARR[pos1]*inj_times;
		  ierr	= VecSetValues(X,1,&pos1,&v,INSERT_VALUES);CHKERRQ(ierr);
		  VecDestroy(&X_SEQ);
		  VecScatterDestroy(&ctx);
		  VecAssemblyBegin(X);
		  VecAssemblyEnd(X);
		  PetscPrintf(MPI_COMM_WORLD,"Inject an error in X of position-%d at the end of iteration-%d\n", pos1,i);
		  flag5	= PETSC_FALSE;
	  }
	  
	  /* Inject two errors in X*/
	  if((i==inj_itr)&&(flag6)&&(error_type==6))
	  {
		  srand (time(NULL));
		  VecScatter	ctx;
		  Vec			X_SEQ;
		  PetscScalar	*X_ARR;
		  VecScatterCreateToAll(X,&ctx,&X_SEQ);
		  VecScatterBegin(ctx,X,X_SEQ,INSERT_VALUES,SCATTER_FORWARD);
		  VecScatterEnd(ctx,X,X_SEQ,INSERT_VALUES,SCATTER_FORWARD);
		  VecGetArray(X_SEQ,&X_ARR);
		  pos1 = rand()/(RAND_MAX/n+1);
		  v		= X_ARR[pos1]*inj_times;
		  ierr	= VecSetValues(X,1,&pos1,&v,INSERT_VALUES);CHKERRQ(ierr);
		  pos2 = rand()/(RAND_MAX/n+1);
		  v		= X_ARR[pos2]*inj_times;
		  ierr	= VecSetValues(X,1,&pos2,&v,INSERT_VALUES);CHKERRQ(ierr);
		  VecDestroy(&X_SEQ);
		  VecScatterDestroy(&ctx);
		  VecAssemblyBegin(X);
		  VecAssemblyEnd(X);
		  PetscPrintf(MPI_COMM_WORLD,"Inject two errors in X of position-%d and position-%d at the end of iteration-%d\n", pos1,pos2,i);
		  flag6	= PETSC_FALSE;
		}
		
		/* Inject an error in R */
	  if((i==inj_itr)&&(flag7)&&(error_type==7))
	  {
		  srand (time(NULL));
		  VecScatter	ctx;
		  Vec			R_SEQ;
		  PetscScalar	*R_ARR;
		  VecScatterCreateToAll(R,&ctx,&R_SEQ);
		  VecScatterBegin(ctx,R,R_SEQ,INSERT_VALUES,SCATTER_FORWARD);
		  VecScatterEnd(ctx,R,R_SEQ,INSERT_VALUES,SCATTER_FORWARD);
		  VecGetArray(R_SEQ,&R_ARR);
		  pos1 = rand()/(RAND_MAX/n+1);
		  v		= R_ARR[pos1]*inj_times;
		  ierr	= VecSetValues(R,1,&pos1,&v,INSERT_VALUES);CHKERRQ(ierr);
		  VecDestroy(&R_SEQ);
		  VecScatterDestroy(&ctx);
		  VecAssemblyBegin(R);
		  VecAssemblyEnd(R);
		  PetscPrintf(MPI_COMM_WORLD,"Inject an error in R of position-%d at the end of iteration-%d\n", pos1,i);
		  flag7	= PETSC_FALSE;
	  }
	  
	  /* Inject two errors */
	  if((i==inj_itr)&&(flag8)&&(error_type==8))
	  {
		  srand (time(NULL));
		  VecScatter	ctx;
		  Vec			R_SEQ;
		  PetscScalar	*R_ARR;
		  VecScatterCreateToAll(R,&ctx,&R_SEQ);
		  VecScatterBegin(ctx,R,R_SEQ,INSERT_VALUES,SCATTER_FORWARD);
		  VecScatterEnd(ctx,R,R_SEQ,INSERT_VALUES,SCATTER_FORWARD);
		  VecGetArray(R_SEQ,&R_ARR);
		  pos1 = rand()/(RAND_MAX/n+1);
		  v		= R_ARR[pos1]*inj_times;
		  ierr	= VecSetValues(R,1,&pos1,&v,INSERT_VALUES);CHKERRQ(ierr);
		  pos2 = rand()/(RAND_MAX/n+1);
		  v		= R_ARR[pos2]*inj_times;
		  ierr	= VecSetValues(R,1,&pos2,&v,INSERT_VALUES);CHKERRQ(ierr);
		  VecDestroy(&R_SEQ);
		  VecScatterDestroy(&ctx);
		  VecAssemblyBegin(R);
		  VecAssemblyEnd(R);
		  PetscPrintf(MPI_COMM_WORLD,"Inject two errors in R of position-%d and position-%d at the end of iteration-%d\n", pos1,pos2,i);
		  flag8	= PETSC_FALSE;
		}
		
		/* Inject an error in Z */
	  if((i==inj_itr)&&(flag9)&&(error_type==9))
	  {
		  srand (time(NULL));
		  VecScatter	ctx;
		  Vec			Z_SEQ;
		  PetscScalar	*Z_ARR;
		  VecScatterCreateToAll(Z,&ctx,&Z_SEQ);
		  VecScatterBegin(ctx,Z,Z_SEQ,INSERT_VALUES,SCATTER_FORWARD);
		  VecScatterEnd(ctx,Z,Z_SEQ,INSERT_VALUES,SCATTER_FORWARD);
		  VecGetArray(Z_SEQ,&Z_ARR);
		  pos1 = rand()/(RAND_MAX/n+1);
		  v		= Z_ARR[pos1]*inj_times;
		  ierr	= VecSetValues(Z,1,&pos1,&v,INSERT_VALUES);CHKERRQ(ierr);
		  VecDestroy(&Z_SEQ);
		  VecScatterDestroy(&ctx);
		  VecAssemblyBegin(Z);
		  VecAssemblyEnd(Z);
		  PetscPrintf(MPI_COMM_WORLD,"Inject an error in Z of position-%d at the end of iteration-%d\n", pos1,i);
		  flag9	= PETSC_FALSE;
	  }
	  
	  /* Inject two errors */
	  if((i==inj_itr)&&(flag10)&&(error_type==10))
	  {
		  srand (time(NULL));
		  VecScatter	ctx;
		  Vec			Z_SEQ;
		  PetscScalar	*Z_ARR;
		  VecScatterCreateToAll(Z,&ctx,&Z_SEQ);
		  VecScatterBegin(ctx,Z,Z_SEQ,INSERT_VALUES,SCATTER_FORWARD);
		  VecScatterEnd(ctx,Z,Z_SEQ,INSERT_VALUES,SCATTER_FORWARD);
		  VecGetArray(Z_SEQ,&Z_ARR);
		  pos1 = rand()/(RAND_MAX/n+1);
		  v		= Z_ARR[pos1]*inj_times;
		  ierr	= VecSetValues(Z,1,&pos1,&v,INSERT_VALUES);CHKERRQ(ierr);
		  pos2 = rand()/(RAND_MAX/n+1);
		  v		= Z_ARR[pos2]*inj_times;
		  ierr	= VecSetValues(Z,1,&pos2,&v,INSERT_VALUES);CHKERRQ(ierr);
		  VecDestroy(&Z_SEQ);
		  VecScatterDestroy(&ctx);
		  VecAssemblyBegin(Z);
		  VecAssemblyEnd(Z);
		  PetscPrintf(MPI_COMM_WORLD,"Inject two errors in Z of position-%d and position-%d at the end of iteration-%d\n", pos1,pos2,i);
		  flag10	= PETSC_FALSE;
		}
	/* Dingwen */
	
  } while (i<ksp->max_it);
  /* Dingwen */
  clock_gettime(CLOCK_REALTIME, &end);
  local_diff = 1000000000L*(end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec);
  MPI_Reduce(&local_diff, &global_diff, 1, MPI_LONG_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  PetscPrintf(MPI_COMM_WORLD,"elapsed time of main loop = %lf nanoseconds\n", (double)(global_diff)/size);
  PetscPrintf(MPI_COMM_WORLD,"Number of iterations without rollback = %d\n", i+1);
  }
  
  if (i >= ksp->max_it) ksp->reason = KSP_DIVERGED_ITS;
  if (eigs) cg->ned = ksp->its;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPDestroy_CG"
PetscErrorCode KSPDestroy_CG(KSP ksp)
{
  KSP_CG         *cg = (KSP_CG*)ksp->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* free space used for singular value calculations */
  if (ksp->calc_sings) {
    ierr = PetscFree4(cg->e,cg->d,cg->ee,cg->dd);CHKERRQ(ierr);
  }
  ierr = KSPDestroyDefault(ksp);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ksp,"KSPCGSetType_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ksp,"KSPCGUseSingleReduction_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
     KSPView_CG - Prints information about the current Krylov method being used

      Currently this only prints information to a file (or stdout) about the
      symmetry of the problem. If your Krylov method has special options or
      flags that information should be printed here.

*/
#undef __FUNCT__
#define __FUNCT__ "KSPView_CG"
PetscErrorCode KSPView_CG(KSP ksp,PetscViewer viewer)
{
#if defined(PETSC_USE_COMPLEX)
  KSP_CG         *cg = (KSP_CG*)ksp->data;
  PetscErrorCode ierr;
  PetscBool      iascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  CG or CGNE: variant %s\n",KSPCGTypes[cg->type]);CHKERRQ(ierr);
  }
#endif
  PetscFunctionReturn(0);
}

/*
    KSPSetFromOptions_CG - Checks the options database for options related to the
                           conjugate gradient method.
*/
#undef __FUNCT__
#define __FUNCT__ "KSPSetFromOptions_CG"
PetscErrorCode KSPSetFromOptions_CG(PetscOptions *PetscOptionsObject,KSP ksp)
{
  PetscErrorCode ierr;
  KSP_CG         *cg = (KSP_CG*)ksp->data;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"KSP CG and CGNE options");CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  ierr = PetscOptionsEnum("-ksp_cg_type","Matrix is Hermitian or complex symmetric","KSPCGSetType",KSPCGTypes,(PetscEnum)cg->type,
                          (PetscEnum*)&cg->type,NULL);CHKERRQ(ierr);
#endif
  ierr = PetscOptionsBool("-ksp_cg_single_reduction","Merge inner products into single MPI_Allreduce()","KSPCGUseSingleReduction",cg->singlereduction,&cg->singlereduction,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
    KSPCGSetType_CG - This is an option that is SPECIFIC to this particular Krylov method.
                      This routine is registered below in KSPCreate_CG() and called from the
                      routine KSPCGSetType() (see the file cgtype.c).
*/
#undef __FUNCT__
#define __FUNCT__ "KSPCGSetType_CG"
static PetscErrorCode  KSPCGSetType_CG(KSP ksp,KSPCGType type)
{
  KSP_CG *cg = (KSP_CG*)ksp->data;

  PetscFunctionBegin;
  cg->type = type;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPCGUseSingleReduction_CG"
static PetscErrorCode  KSPCGUseSingleReduction_CG(KSP ksp,PetscBool flg)
{
  KSP_CG *cg = (KSP_CG*)ksp->data;

  PetscFunctionBegin;
  cg->singlereduction = flg;
  PetscFunctionReturn(0);
}

/*
    KSPCreate_CG - Creates the data structure for the Krylov method CG and sets the
       function pointers for all the routines it needs to call (KSPSolve_CG() etc)

    It must be labeled as PETSC_EXTERN to be dynamically linkable in C++
*/
/*MC
     KSPCG - The preconditioned conjugate gradient (PCG) iterative method

   Options Database Keys:
+   -ksp_cg_type Hermitian - (for complex matrices only) indicates the matrix is Hermitian, see KSPCGSetType()
.   -ksp_cg_type symmetric - (for complex matrices only) indicates the matrix is symmetric
-   -ksp_cg_single_reduction - performs both inner products needed in the algorithm with a single MPI_Allreduce() call, see KSPCGUseSingleReduction()

   Level: beginner

   Notes: The PCG method requires both the matrix and preconditioner to be symmetric positive (or negative) (semi) definite
          Only left preconditioning is supported.

   For complex numbers there are two different CG methods. One for Hermitian symmetric matrices and one for non-Hermitian symmetric matrices. Use
   KSPCGSetType() to indicate which type you are using.

   Developer Notes: KSPSolve_CG() should actually query the matrix to determine if it is Hermitian symmetric or not and NOT require the user to
   indicate it to the KSP object.

   References:
   Methods of Conjugate Gradients for Solving Linear Systems, Magnus R. Hestenes and Eduard Stiefel,
   Journal of Research of the National Bureau of Standards Vol. 49, No. 6, December 1952 Research Paper 2379
   pp. 409--436.

.seealso:  KSPCreate(), KSPSetType(), KSPType (for list of available types), KSP,
           KSPCGSetType(), KSPCGUseSingleReduction(), KSPPIPECG, KSPGROPPCG

M*/
#undef __FUNCT__
#define __FUNCT__ "KSPCreate_CG"
PETSC_EXTERN PetscErrorCode KSPCreate_CG(KSP ksp)
{
  PetscErrorCode ierr;
  KSP_CG         *cg;

  PetscFunctionBegin;
  ierr = PetscNewLog(ksp,&cg);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
  cg->type = KSP_CG_SYMMETRIC;
#else
  cg->type = KSP_CG_HERMITIAN;
#endif
  ksp->data = (void*)cg;

  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_PRECONDITIONED,PC_LEFT,3);CHKERRQ(ierr);
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_UNPRECONDITIONED,PC_LEFT,2);CHKERRQ(ierr);
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_NATURAL,PC_LEFT,2);CHKERRQ(ierr);
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_NONE,PC_LEFT,2);CHKERRQ(ierr);

  /*
       Sets the functions that are associated with this data structure
       (in C++ this is the same as defining virtual functions)
  */
  ksp->ops->setup          = KSPSetUp_CG;
  ksp->ops->solve          = KSPSolve_CG;
  ksp->ops->destroy        = KSPDestroy_CG;
  ksp->ops->view           = KSPView_CG;
  ksp->ops->setfromoptions = KSPSetFromOptions_CG;
  ksp->ops->buildsolution  = KSPBuildSolutionDefault;
  ksp->ops->buildresidual  = KSPBuildResidualDefault;

  /*
      Attach the function KSPCGSetType_CG() to this object. The routine
      KSPCGSetType() checks for this attached function and calls it if it finds
      it. (Sort of like a dynamic member function that can be added at run time
  */
  ierr = PetscObjectComposeFunction((PetscObject)ksp,"KSPCGSetType_C",KSPCGSetType_CG);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ksp,"KSPCGUseSingleReduction_C",KSPCGUseSingleReduction_CG);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}