/**
 * @file c_utils.h
 * @brief header file for c_utils.c providing C API for other C extensions.
 */

#ifndef C_UTILS_H
#define C_UTILS_H

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#ifndef PY_SSIZE_T_CLEAN
#define PY_SSIZE_T_CLEAN
#include "Python.h"
#endif /* PY_SSIZE_T_CLEAN */

// number of pointers in the C API
#define PyFastMin_UtilsAPI_Len 1
// index, return, and prototype for PyFastMin_KwargsTrim. style is copied from
// https://docs.python.org/3/extending/extending.html
#define PyFastMin_KwargsTrim_NUM 0

// if C_UTILS_MODULE defined, has been included in c_utils.c
#ifdef C_UTILS_MODULE
// mark any functions that can only be used from C code
#define C_API_ONLY(name) name
#else
// declare the void ** for the c_utils C API
static void **PyFastMin_UtilsAPI;
// macros for accessing the stored function pointers
#define PyFastMin_KwargsTrim (*(PyObject *(*)(PyObject *, char ** const))) \
  PyFastMin_UtilsAPI[PyFastMin_KwargsTrim_NUM])

/**
 * Importing function returning -1 on error and 0 on success.
 */
static int
import_utils_CAPI(void)
{
  // import into module as _C_API and return exit status
  PyFastMin_UtilsAPI = (void **) PyCapsule_Import("c_utils._C_API", 0);
  return (PyFastMin_UtilsAPI != NULL) ? 0 : -1;
}

#endif /* C_UTILS_MODULE */

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* C_UTILS_H */