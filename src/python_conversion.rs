//! Convert native samples to Python objects, preserving Python exceptions.

use crate::config::ImageFormat;
use crate::image_payload::PythonImagePayload;
use crate::structs::Sample;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};

/// Convert an image payload to the requested Python representation.
fn image_to_python(
    payload: &PythonImagePayload,
    py: Python<'_>,
    image_format: ImageFormat,
) -> PyResult<Py<PyAny>> {
    match image_format {
        ImageFormat::Pil => payload.to_pil_image(py),
        ImageFormat::Numpy => payload.to_numpy_array(py),
    }
}

pub fn sample_to_python_types(
    sample: Sample,
    py: Python<'_>,
    image_format: ImageFormat,
) -> PyResult<Py<PyAny>> {
    // Convert the sample to a Python dict, using PIL images or zero-copy arrays
    // depending on the requested format.
    let sample_dict = PyDict::new(py);

    // Add basic fields
    sample_dict.set_item("id", sample.id)?;
    sample_dict.set_item("source", sample.source)?;
    sample_dict.set_item("duplicate_state", sample.duplicate_state)?;

    // Convert attributes to python dict
    let attributes_dict = PyDict::new(py);
    for (key, value) in sample.attributes {
        // String values are passed through as-is; everything else is JSON-encoded.
        if let Some(text) = value.as_str() {
            attributes_dict.set_item(key, text)?;
        } else {
            let encoded = serde_json::to_string(&value).unwrap_or_else(|_| "{}".to_string());
            attributes_dict.set_item(key, encoded)?;
        }
    }
    sample_dict.set_item("attributes", attributes_dict)?;

    // Convert tags
    let tags_list = PyList::new(py, &sample.tags)?;
    sample_dict.set_item("tags", tags_list)?;

    // Convert coca_embedding
    let coca_array = PyList::new(py, &sample.coca_embedding)?;
    sample_dict.set_item("coca_embedding", coca_array)?;

    // Convert latents
    let latents_dict = PyDict::new(py);
    for (key, latent) in sample.latents {
        let latent_dict = PyDict::new(py);
        latent_dict.set_item("data", PyBytes::new(py, &latent.data))?;
        latent_dict.set_item("len", latent.len)?;
        latents_dict.set_item(key, latent_dict)?;
    }
    sample_dict.set_item("latents", latents_dict)?;

    // Convert the main, mask and additional images to the requested format.
    let image = image_to_python(&sample.image, py, image_format)?;
    sample_dict.set_item("image", image)?;

    let masks_dict = PyDict::new(py);
    for (key, mask) in sample.masks {
        let mask_obj = image_to_python(&mask, py, image_format)?;
        masks_dict.set_item(key, mask_obj)?;
    }
    sample_dict.set_item("masks", masks_dict)?;

    let additional_images_dict = PyDict::new(py);
    for (key, additional_image) in sample.additional_images {
        let additional_obj = image_to_python(&additional_image, py, image_format)?;
        additional_images_dict.set_item(key, additional_obj)?;
    }
    sample_dict.set_item("additional_images", additional_images_dict)?;

    Ok(sample_dict.into())
}
