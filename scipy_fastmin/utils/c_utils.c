/**
 * @file c_utils.c
 * @brief C extension module with utilities for other C extension modules.
 */

#define PY_SSIZE_T_CLEAN
#include "Python.h"

#include <math.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"

#define C_UTILS_MODULE
#include "c_utils.h"

PyDoc_STRVAR(
  module_doc,
  "C extension module containing utilities for other C extension modules."
  "\n\n"
  "Both Python API and C API functions are included in the module. Access to\n"
  "the C API is granted by including c_utils.h and calling import_c_utils in\n"
  "the PyInit_* method of the respective C extension module."
);

/**
 * Return new kwargs containing only the pairs whose keys are in `keep_list`.
 * 
 * References of internal objects in `dict` are not touched.
 * 
 * @param dict `PyObject *` dict whose keys are all string keys
 * @param keep_list `char ** const` `NULL`-terminated array of keys to keep.
 * @returns New `PyObject *` dict only containing keys in `keep_list`. `NULL`
 *     returned on error, for example if `dict` is not a dict.
 */
static PyObject *
C_API_ONLY(PyFastMin_KwargsTrim)(PyObject *dict, char ** const keep_list)
{
  // if not a dict, error
  if (!PyDict_Check(dict)) {
    return NULL;
  }
  // if keep_list is NULL, error
  if (keep_list == NULL) {
    PyErr_SetString(PyExc_TypeError, "keep_list is NULL");
    return NULL;
  }
  // new dict that will contain only keys in keep_list
  PyObject *new_dict = PyDict_New();
  if (new_dict == NULL) {
    return NULL;
  }
  // for all keys in keep_list
  int i = 0;
  while (keep_list[i] != NULL) {
    // create string key from keep_list[i]. if NULL, error
    PyObject *key = PyUnicode_FromString(keep_list[i]);
    if (key == NULL) {
      return NULL;
    }
    // if key is in dict, set new key-value pair to new_dict
    if (PyDict_Contains(dict, key)) {
      // get borrowed reference from dict. if NULL, error, as key is supposed
      // to already be in dict. exception will be raised.
      PyObject *val = PyDict_GetItemWithError(dict, key);
      if (val == NULL) {
        Py_DECREF(key);
        Py_DECREF(new_dict);
        return NULL;
      }
      // else set value to new_dict as well
      if (PyDict_SetItemString(new_dict, keep_list[i], val) < 0) {
        Py_DECREF(key);
        Py_DECREF(new_dict);
      }
    }
    // clean up key and move on
    Py_DECREF(key);
    i++;
  }
  // done, so return new_dict
  return new_dict;
}

/**
 * Returns the 2-norm of an aligned C-contiguous ndarray with type NPY_DOUBLE.
 * 
 * If ar does not have NPY_DOUBLE typenum/is not aligned, a new array will be
 * created. Returns -1 on error since norms cannot be negative. Note that we
 * specify `PyObject *` instead of `PyArrayObject *` in order to avoid the
 * annoying `NO_IMPORT_ARRAY` `PY_ARRAY_UNIQUE_SYMBOL` headaches.
 * 
 * @param ar `PyObject *` that is C contiguous
 * @returns The 2-norm of `ar` as a C `double`, -1 on error.
 */
static double
C_API_ONLY(PyFastMin_PyArrayNorm)(PyArrayObject *ar) {
  // ar must be ndarray
  if (!PyArray_Check(ar)) {
    PyErr_SetString(PyExc_TypeError, "ar must be a numpy.ndarray");
    return -1;
  }
  // if ar has type NPY_DOUBLE and is aligned, Py_INCREF it (will Py_DECREF
  // later in function). note that we don't care about order.
  if ((PyArray_TYPE(ar) == NPY_DOUBLE) && PyArray_ISALIGNED(ar)) {
    Py_INCREF(ar);
  }
  // else drop borrowed reference and create new ndarray. order is irrelevant.
  else {
    ar = (PyArrayObject *) PyArray_FromArray(
      ar, PyArray_DescrNewFromType(NPY_DOUBLE), NPY_ARRAY_CARRAY
    );
  }
  // get data member
  double *data = (double *) PyArray_DATA(ar);
  // initial value of the norm
  double norm = 0;
  // compute value of the norm
  for (npy_intp i = 0; i < PyArray_SIZE(ar); i++) {
    norm = norm + data[i] * data[i];
  }
  norm = sqrt(norm);
  // Py_DECREF ar (guaranteed to either be new or Py_INCREF'd) and return
  Py_DECREF(ar);
  return norm;
}

// method table
static PyMethodDef mod_methods[] = {
  {NULL, NULL, 0, NULL}
};

// static module definition
static PyModuleDef mod_struct = {
  PyModuleDef_HEAD_INIT,
  "c_utils",
  module_doc,
  -1,
  mod_methods
};

// module initialization function (external linkage)
PyMODINIT_FUNC PyInit_c_utils(void)
{
  // import_array sets error indicator and returns NULL on error automatically
  import_array();
  // create module; if NULL, error
  PyObject *module = PyModule_Create(&mod_struct);
  if (module == NULL) {
    return NULL;
  }
  // initialize static array of void * holding the C API
  static void *utils_api[PyFastMin_UtilsAPI_Len];
  utils_api[PyFastMin_KwargsTrim_NUM] = (void *) PyFastMin_KwargsTrim;
  utils_api[PyFastMin_PyArrayNorm_NUM] = (void *) PyFastMin_PyArrayNorm;
  // create capsule holding C API void **. if NULL, error
  PyObject *api_capsule = PyCapsule_New(
    (void *) utils_api, "c_utils._C_API", NULL
  );
  if (api_capsule == NULL) {
    Py_DECREF(module);
    return NULL;
  }
  // try to add capsule to module. since PyModule_AddObject steals reference
  // only on success, we need to Py_DECREF api_capsule on failure.
  if (PyModule_AddObject(module, "_C_API", api_capsule) < 0) {
    Py_DECREF(api_capsule);
    Py_DECREF(module);
    return NULL;
  }
  // return completed module
  return module;
}