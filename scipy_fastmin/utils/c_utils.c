/**
 * @file c_utils.c
 * @brief C extension module with utilities for other C extension modules.
 */

#define PY_SSIZE_T_CLEAN
#include "Python.h"

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
    PyErr_SetString(PyExc_TypeError, "dict must be of type dict");
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
      PyErr_SetString(
        PyExc_RuntimeError, "could not construct key from keep_list[i]"
      );
      return NULL;
    }
    // if key is in dict, set new key-value pair to new_dict
    if (PyDict_Contains(dict, key)) {
      // get borrowed reference from dict. if NULL, error, as key is supposed
      // to already be in dict. exception will be raised.
      PyObject *val = PyDict_GetItemWithError(dict, key);
      if (val == NULL) {
        PyErr_SetString(
          PyExc_RuntimeError, "key inexplicably missing from dict"
        );
        Py_DECREF(key);
        Py_DECREF(new_dict);
        return NULL;
      }
      // else set value to new_dict as well
      if (PyDict_SetItemString(new_dict, keep_list[i], val) < 0) {
        PyErr_SetString(
          PyExc_RuntimeError, "failed to assign value to new_dict"
        );
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
  // create module; if NULL, error
  PyObject *module = PyModule_Create(&mod_struct);
  if (module == NULL) {
    PyErr_SetString(PyExc_RuntimeError, "FATAL: module creation failed");
    return NULL;
  }
  // initialize static array of void * holding the C API
  static void *PyFastMin_UtilsAPI[PyFastMin_UtilsAPI_Len];
  PyFastMin_UtilsAPI[PyFastMin_KwargsTrim_NUM] = (void *) PyFastMin_KwargsTrim;
  // create capsule holding C API void **. if NULL, error
  PyObject *api_capsule = PyCapsule_New(
    (void *) PyFastMin_UtilsAPI, "c_utils._C_API", NULL
  );
  if (api_capsule == NULL) {
    PyErr_SetString(PyExc_RuntimeError, "FATAL: capsule creation failed");
    Py_DECREF(module);
    return NULL;
  }
  // try to add capsule to module. since PyModule_AddObject steals reference
  // only on success, we need to Py_DECREF api_capsule on failure.
  if (PyModule_AddObject(module, "_C_API", api_capsule) < 0) {
    PyErr_SetString(
      PyExc_RuntimeError, "FATAL: could not add capsule to module"
    );
    Py_DECREF(api_capsule);
    Py_DECREF(module);
    return NULL;
  }
  // return completed module
  return module;
}