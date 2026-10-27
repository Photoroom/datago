//! Read-only byte exports retain an immutable allocation owner in Py_buffer.obj.

use pyo3::exceptions::{PyBufferError, PyOverflowError};
use pyo3::ffi;
use pyo3::prelude::*;
use std::ffi::c_int;
use std::sync::Arc;

/// Separate from the mutable payload wrapper: reassignment of its data cannot
/// invalidate outstanding memoryviews or NumPy arrays.
#[pyclass(frozen)]
pub(crate) struct SharedImageBuffer {
    pub(crate) data: Arc<Vec<u8>>,
}

#[pymethods]
impl SharedImageBuffer {
    unsafe fn __getbuffer__(
        slf: Bound<'_, Self>,
        view: *mut ffi::Py_buffer,
        flags: c_int,
    ) -> PyResult<()> {
        // SAFETY: PyO3 supplies CPython's writable buffer descriptor.
        unsafe { fill_readonly_buffer(slf, view, flags) }
    }
}

/// Export a snapshot of the allocation, independent of the payload's lifetime.
/// Safety: `view` must point to a writable CPython buffer descriptor.
pub(crate) unsafe fn export_bytes(
    py: Python<'_>,
    data: Arc<Vec<u8>>,
    view: *mut ffi::Py_buffer,
    flags: c_int,
) -> PyResult<()> {
    let owner = Bound::new(py, SharedImageBuffer { data })?;
    // SAFETY: the caller supplies a valid descriptor; the owner retains the data.
    unsafe { fill_readonly_buffer(owner, view, flags) }
}

unsafe fn fill_readonly_buffer(
    owner: Bound<'_, SharedImageBuffer>,
    view: *mut ffi::Py_buffer,
    flags: c_int,
) -> PyResult<()> {
    if view.is_null() {
        return Err(PyBufferError::new_err("buffer view is null"));
    }
    let data = &owner.get().data;
    let len = isize::try_from(data.len())
        .map_err(|_| PyOverflowError::new_err("image buffer exceeds Python's size limit"))?;
    // SAFETY: `view` is writable and `owner` retains the immutable allocation.
    // CPython fills the byte-buffer metadata, rejects writable requests, and
    // increments the owner's reference count until PyBuffer_Release is called.
    let result = unsafe {
        ffi::PyBuffer_FillInfo(
            view,
            owner.as_ptr(),
            data.as_ptr().cast_mut().cast(),
            len,
            1,
            flags,
        )
    };
    if result == -1 {
        Err(PyErr::fetch(owner.py()))
    } else {
        Ok(())
    }
}
