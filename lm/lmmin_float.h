/*
 * Project:  LevenbergMarquardtLeastSquaresFitting
 *
 * File:     lmmin.h
 *
 * Contents: Public interface to the Levenberg-Marquardt core implementation.
 *
 * Author:   Joachim Wuttke 2004-2010
 * 
 * Homepage: www.messen-und-deuten.de/lmfit
 */

#ifndef LMMIN_FLOAT_H
#define LMMIN_FLOAT_H


namespace lmmin_float
{

/** Compact high-level interface. **/

/* Collection of control (input) parameters. */
typedef struct {
    float ftol;      /* relative error desired in the sum of squares. */
    float xtol;      /* relative error between last two approximations. */
    float gtol;      /* orthogonality desired between fvec and its derivs. */
    float epsilon;   /* step used to calculate the jacobian. */
    float stepbound; /* initial bound to steps in the outer loop. */
    int maxcall;     /* maximum number of iterations. */
    int scale_diag;  /* UNDOCUMENTED, TESTWISE automatical diag rescaling? */
    int printflags;  /* OR'ed to produce more noise */
} lm_control_struct;

/* Collection of status (output) parameters. */
typedef struct {
    float fnorm;     /* norm of the residue vector fvec. */
    int nfev;	     /* actual number of iterations. */
    int info;	     /* status of minimization. */
} lm_status_struct;

/* Recommended control parameter settings. */
extern const lm_control_struct lm_control_double;
extern const lm_control_struct lm_control_float;

/* Standard monitoring routine. */
void lm_printout_std( int n_par, const float *par, int m_dat,
                     const void *data, const float *fvec,
                     int printflags, int iflag, int iter, int nfev);

/* Refined calculation of Eucledian norm, typically used in printout routine. */
float lm_enorm( int, const float * );

/* The actual minimization. */
void lmmin( int n_par, float *par, int m_dat, const void *data,
           void (*evaluate) (const float *par, int m_dat, const void *data,
                             float *fvec, int *info),
           const lm_control_struct *control, lm_status_struct *status,
           void (*printout) (int n_par, const float *par, int m_dat,
                             const void *data, const float *fvec,
                             int printflags, int iflag, int iter, int nfev) );


/** Legacy low-level interface. **/

/* Alternative to lm_minimize, allowing full control, and read-out
 of auxiliary arrays. For usage, see implementation of lm_minimize. */
void lm_lmdif( int m, int n, float *x, float *fvec, float ftol,
              float xtol, float gtol, int maxfev, float epsfcn,
              float *diag, int mode, float factor, int *info, int *nfev,
              float *fjac, int *ipvt, float *qtf, float *wa1,
              float *wa2, float *wa3, float *wa4,
              void (*evaluate) (const float *par, int m_dat, const void *data,
                                float *fvec, int *info),
              void (*printout) (int n_par, const float *par, int m_dat,
                                const void *data, const float *fvec,
                                int printflags, int iflag, int iter, int nfev),
              int printflags, const void *data );

extern const char *lm_infmsg[];
extern const char *lm_shortmsg[];

} // lmmin_float

#endif /* LMMIN_FLOAT_H */
