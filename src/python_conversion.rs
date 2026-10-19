use crate::structs::Sample;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};

pub fn sample_to_python_types(sample: Sample, py: Python<'_>) -> Option<Py<PyAny>> {
    // Convert the sample to a Python dict with PIL images
    let sample_dict = PyDict::new(py);

    // Add basic fields
    sample_dict.set_item("id", sample.id).unwrap();
    sample_dict.set_item("source", sample.source).unwrap();
    sample_dict
        .set_item("duplicate_state", sample.duplicate_state)
        .unwrap();

    // Convert attributes to python dict
    let attributes_dict = PyDict::new(py);
    for (key, value) in sample.attributes {
        // If value is a string it can be passed as is, else we pass the json-encoded version
        let value_serialized: String = if value.is_string() {
            value.to_string()
        } else {
            serde_json::to_string(&value).unwrap_or("{}".to_string())
        };
        attributes_dict.set_item(key, value_serialized).unwrap();
    }
    sample_dict.set_item("attributes", attributes_dict).unwrap();

    // Convert tags
    let tags_list = PyList::new(py, &sample.tags).unwrap();
    sample_dict.set_item("tags", tags_list).unwrap();

    // Convert coca_embedding
    let coca_array = PyList::new(py, &sample.coca_embedding).unwrap();
    sample_dict.set_item("coca_embedding", coca_array).unwrap();

    // Convert latents
    let latents_dict = PyDict::new(py);
    for (key, latent) in sample.latents {
        let latent_dict = PyDict::new(py);
        latent_dict
            .set_item("data", PyBytes::new(py, &latent.data))
            .unwrap();
        latent_dict.set_item("len", latent.len).unwrap();
        latents_dict.set_item(key, latent_dict).unwrap();
    }
    sample_dict.set_item("latents", latents_dict).unwrap();

    // Convert images to PIL images
    let image_pil = sample.image.to_pil_image(py).unwrap();
    sample_dict.set_item("image", image_pil).unwrap();

    // Convert masks to PIL images
    let masks_dict = PyDict::new(py);
    for (key, mask) in sample.masks {
        let mask_pil = mask.to_pil_image(py).unwrap();
        masks_dict.set_item(key, mask_pil).unwrap();
    }
    sample_dict.set_item("masks", masks_dict).unwrap();

    // Convert additional images to PIL images
    let additional_images_dict = PyDict::new(py);
    for (key, additional_image) in sample.additional_images {
        let additional_image_pil = additional_image.to_pil_image(py).unwrap();
        additional_images_dict
            .set_item(key, additional_image_pil)
            .unwrap();
    }
    sample_dict
        .set_item("additional_images", additional_images_dict)
        .unwrap();

    Some(sample_dict.into())
}
