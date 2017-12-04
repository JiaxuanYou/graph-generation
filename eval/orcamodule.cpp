#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <Python.h>

#include "orca/orca.h"

static PyObject *
orca_motifs(PyObject *self, PyObject *args)
{
    const char *orbit_type;
    int graphlet_size;
    const char *input_filename;
    const char *output_filename;
    int sts;

    if (!PyArg_ParseTuple(args, "siss", &orbit_type, &graphlet_size, &input_filename, &output_filename))
        return NULL;
    sts = system(orbit_type);
    motif_counts(orbit_type, graphlet_size, input_filename, output_filename);
    return PyLong_FromLong(sts);
}

static PyMethodDef OrcaMethods[] = {
    {"motifs",  orca_motifs, METH_VARARGS,
     "Compute motif counts."},
};

static struct PyModuleDef orcamodule = {
   PyModuleDef_HEAD_INIT,
   "orca",   /* name of module */
   NULL, /* module documentation, may be NULL */
   -1,       /* size of per-interpreter state of the module,
                or -1 if the module keeps state in global variables. */
   OrcaMethods
};

PyMODINIT_FUNC
PyInit_orca(void)
{
    return PyModule_Create(&orcamodule);
}

int main(int argc, char *argv[]) {

    wchar_t *program = Py_DecodeLocale(argv[0], NULL);
    if (program == NULL) {
        fprintf(stderr, "Fatal error: cannot decode argv[0]\n");
        exit(1);
    }

    /* Add a built-in module, before Py_Initialize */
    PyImport_AppendInittab("orca", PyInit_orca);

    /* Pass argv[0] to the Python interpreter */
    Py_SetProgramName(program);

    /* Initialize the Python interpreter.  Required. */
    Py_Initialize();

    /* Optionally import the module; alternatively,
       import can be deferred until the embedded script
       imports it. */
    PyImport_ImportModule("orca");

    PyMem_RawFree(program);

}

