//! Read-only byte export. This module owns the Python buffer protocol boundary.

use pyo3::exceptions::PyBufferError;
use pyo3::ffi;
use pyo3::prelude::*;
use std::ffi::{c_int, c_void};
use std::ptr;
use std::sync::Arc;

/// Internal buffer owner used when callers start from the Rust ImagePayload
/// pyclass instead of PythonImagePayload. Its Arc is retained by NumPy's base.
/// The wrapped bytes are raw pixels or an encoded stream, mirroring the payload.
#[pyclass]
pub(crate) struct SharedImageBuffer {
    pub(crate) data: Arc<[u8]>,
}

#[pymethods]
impl SharedImageBuffer {
    /// Export raw pixels or an encoded stream as immutable bytes.
    /// Safety: CPython supplies a valid view; the Arc owner survives its release.
    unsafe fn __getbuffer__(
        slf: Bound<'_, Self>,
        view: *mut ffi::Py_buffer,
        flags: c_int,
    ) -> PyResult<()> {
        let owner = slf.borrow();
        unsafe { fill_readonly_buffer(view, flags, owner.data.as_ref(), slf.into_any()) }
    }
}

/// The view retains its own Arc owner, so reassignment of a payload's data cannot
/// invalidate an existing export. The caller must supply CPython's writable view.
pub(crate) unsafe fn export_bytes(
    py: Python<'_>,
    data: Arc<[u8]>,
    view: *mut ffi::Py_buffer,
    flags: c_int,
) -> PyResult<()> {
    let owner = Py::new(py, SharedImageBuffer { data: data.clone() })?;
    // SAFETY: the owner retains exactly the allocation whose pointer is exported.
    unsafe { fill_readonly_buffer(view, flags, &data, owner.into_bound(py).into_any()) }
}

/// Export a read-only one-dimensional byte buffer. The Python owner is retained
/// in Py_buffer.obj, so its Arc-backed data stays alive for every outstanding view.
unsafe fn fill_readonly_buffer(
    view: *mut ffi::Py_buffer,
    flags: c_int,
    data: &[u8],
    owner: Bound<'_, PyAny>,
) -> PyResult<()> {
    if view.is_null() {
        return Err(PyBufferError::new_err("buffer view is null"));
    }
    if flags & ffi::PyBUF_WRITABLE == ffi::PyBUF_WRITABLE {
        return Err(PyBufferError::new_err("image payload buffer is read-only"));
    }
    // SAFETY: the caller supplies a writable Py_buffer. `owner` owns an Arc to
    // `data`; transferring its reference into obj keeps the bytes alive until
    // PyBuffer_Release decrefs it. No consumer may write into this allocation.
    // Format is static; shape and stride refer to fields in the supplied view,
    // following the same convention as PyBuffer_FillInfo for a byte buffer.
    unsafe {
        let buffer = &mut *view;
        buffer.obj = owner.into_ptr();
        buffer.buf = data.as_ptr() as *mut c_void;
        buffer.len = data.len() as isize;
        buffer.readonly = 1;
        buffer.itemsize = 1;
        buffer.format = if flags & ffi::PyBUF_FORMAT == ffi::PyBUF_FORMAT {
            c"B".as_ptr().cast_mut()
        } else {
            ptr::null_mut()
        };
        buffer.ndim = 1;
        buffer.shape = if flags & ffi::PyBUF_ND == ffi::PyBUF_ND {
            ptr::addr_of_mut!((*view).len)
        } else {
            ptr::null_mut()
        };
        buffer.strides = if flags & ffi::PyBUF_STRIDES == ffi::PyBUF_STRIDES {
            ptr::addr_of_mut!((*view).itemsize)
        } else {
            ptr::null_mut()
        };
        buffer.suboffsets = ptr::null_mut();
        buffer.internal = ptr::null_mut();
    }
    Ok(())
}
