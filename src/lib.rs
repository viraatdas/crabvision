use image::{imageops::FilterType, DynamicImage, GenericImageView};
use numpy::ndarray::{ArrayD, ArrayViewD, IxDyn};
use numpy::{
    IntoPyArray, PyArrayDyn, PyReadwriteArrayDyn, PyReadonlyArrayDyn, PyUntypedArrayMethods,
};
use numpy::PyArrayMethods;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyByteArray, PyBytes};
use pyo3::types::{PyList, PySequence};
use std::io::Cursor;
use std::path::PathBuf;

// A tiny, safe subset of cv2 in Rust.
// Goals: imread, imwrite, cvtColor (basic), resize with familiar constants.

// OpenCV-like constants (subset)
const IMREAD_GRAYSCALE: i32 = 0;
const IMREAD_COLOR: i32 = 1;
const IMREAD_UNCHANGED: i32 = -1;

const COLOR_BGR2RGB: i32 = 4; // matches OpenCV numeric
const COLOR_RGB2BGR: i32 = 4; // same value in OpenCV
const COLOR_BGR2GRAY: i32 = 6;
const COLOR_GRAY2BGR: i32 = 8;

const INTER_NEAREST: i32 = 0;
const INTER_LINEAR: i32 = 1;
const INTER_CUBIC: i32 = 2;
const INTER_AREA: i32 = 3; // approximated to Gaussian below
const INTER_LANCZOS4: i32 = 4;

const NORM_INF: i32 = 1;
const NORM_L1: i32 = 2;
const NORM_L2: i32 = 4;

const THRESH_BINARY: i32 = 0;
const THRESH_BINARY_INV: i32 = 1;
const THRESH_TRUNC: i32 = 2;
const THRESH_TOZERO: i32 = 3;
const THRESH_TOZERO_INV: i32 = 4;

const CMP_EQ: i32 = 0;
const CMP_GT: i32 = 1;
const CMP_GE: i32 = 2;
const CMP_LT: i32 = 3;
const CMP_LE: i32 = 4;
const CMP_NE: i32 = 5;

const BORDER_DEFAULT: i32 = 4;

const CV_8U: i32 = 0;
const CV_16S: i32 = 3;
const CV_32F: i32 = 5;
const CV_64F: i32 = 6;

const ROTATE_90_CLOCKWISE: i32 = 0;
const ROTATE_180: i32 = 1;
const ROTATE_90_COUNTERCLOCKWISE: i32 = 2;

fn scalar_u8_from_any(obj: &Bound<'_, PyAny>) -> PyResult<u8> {
    if let Ok(v) = obj.extract::<u8>() {
        return Ok(v);
    }
    if let Ok(v) = obj.extract::<i64>() {
        return Ok(v.clamp(0, 255) as u8);
    }
    if let Ok(v) = obj.extract::<f64>() {
        if !v.is_finite() {
            return Err(PyValueError::new_err("value must be finite"));
        }
        return Ok(v.round().clamp(0.0, 255.0) as u8);
    }
    Err(PyTypeError::new_err("expected an integer-like scalar"))
}

fn vec_u8_from_scalar_or_seq(obj: &Bound<'_, PyAny>, len: usize) -> PyResult<Vec<u8>> {
    if len == 0 {
        return Ok(vec![]);
    }

    if let Ok(v) = scalar_u8_from_any(obj) {
        return Ok(vec![v; len]);
    }

    if let Ok(arr) = obj.extract::<PyReadonlyArrayDyn<'_, u8>>() {
        let flat = arr.as_array().iter().copied().collect::<Vec<u8>>();
        if flat.len() == len {
            return Ok(flat);
        }
        return Err(PyValueError::new_err(format!(
            "expected {len} values, got {}",
            flat.len()
        )));
    }

    let seq = obj
        .downcast::<PySequence>()
        .map_err(|_| PyTypeError::new_err("expected a scalar or a sequence"))?;
    if seq.len()? as usize != len {
        return Err(PyValueError::new_err(format!(
            "expected {len} values, got {}",
            seq.len()?
        )));
    }
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        out.push(scalar_u8_from_any(&seq.get_item(i)?)?);
    }
    Ok(out)
}

fn filter_from_interpolation(inter: i32) -> FilterType {
    match inter {
        INTER_NEAREST => FilterType::Nearest,
        INTER_LINEAR => FilterType::Triangle,
        INTER_CUBIC => FilterType::CatmullRom,
        INTER_AREA => FilterType::Gaussian,
        INTER_LANCZOS4 => FilterType::Lanczos3,
        _ => FilterType::Triangle,
    }
}

fn rgb_to_bgr_inplace(buf: &mut [u8]) {
    for chunk in buf.chunks_mut(3) {
        chunk.swap(0, 2);
    }
}

fn gray_to_bgr(buf: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(buf.len() * 3);
    for &px in buf {
        out.extend_from_slice(&[px, px, px]);
    }
    out
}

fn bgr_to_gray(buf: &[u8]) -> Vec<u8> {
    // Use luma approximation: 0.114*B + 0.587*G + 0.299*R
    let mut out = Vec::with_capacity(buf.len() / 3);
    for chunk in buf.chunks(3) {
        let b = chunk[0] as f32;
        let g = chunk[1] as f32;
        let r = chunk[2] as f32;
        let y = 0.114 * b + 0.587 * g + 0.299 * r;
        out.push(y.round().clamp(0.0, 255.0) as u8);
    }
    out
}

fn pyarray_from_vec(py: Python<'_>, data: Vec<u8>, shape: &[usize]) -> PyResult<Py<PyArrayDyn<u8>>> {
    let arr = ArrayD::from_shape_vec(IxDyn(shape), data)
        .map_err(|e| PyValueError::new_err(format!("failed to create ndarray: {e}")))?;
    let py_arr = arr.into_pyarray_bound(py);
    Ok(py_arr.unbind())
}

fn decode_image_bytes(py: Python<'_>, bytes: &[u8], flags: i32) -> PyResult<Py<PyArrayDyn<u8>>> {
    let img = image::load_from_memory(bytes)
        .map_err(|e| PyValueError::new_err(format!("failed to decode image: {e}")))?;
    let (w, h) = img.dimensions();

    // For now, treat IMREAD_UNCHANGED as IMREAD_COLOR (no alpha support yet).
    let effective_flags = if flags == IMREAD_UNCHANGED {
        IMREAD_COLOR
    } else {
        flags
    };

    match effective_flags {
        IMREAD_GRAYSCALE => {
            let gray = img.to_luma8();
            pyarray_from_vec(py, gray.into_raw(), &[h as usize, w as usize])
        }
        _ => {
            let mut rgb = match img {
                DynamicImage::ImageRgb8(i) => i.into_raw(),
                _ => img.to_rgb8().into_raw(),
            };
            rgb_to_bgr_inplace(&mut rgb);
            pyarray_from_vec(py, rgb, &[h as usize, w as usize, 3])
        }
    }
}

fn any_to_bytes(obj: &Bound<'_, PyAny>) -> PyResult<Vec<u8>> {
    if let Ok(b) = obj.downcast::<PyBytes>() {
        Ok(b.as_bytes().to_vec())
    } else if let Ok(b) = obj.downcast::<PyByteArray>() {
        Ok(b.to_vec())
    } else if let Ok(arr) = obj.extract::<PyReadonlyArrayDyn<'_, u8>>() {
        let shape = arr.shape().to_vec();
        if shape.len() != 1 {
            return Err(PyValueError::new_err("expected 1D uint8 buffer for image bytes"));
        }
        Ok(arr.as_array().to_owned().into_raw_vec())
    } else {
        Err(PyTypeError::new_err(
            "expected bytes, bytearray, or 1D np.ndarray[uint8]",
        ))
    }
}

fn py_image_ndarray_to_vec_and_shape(arr: &PyReadonlyArrayDyn<u8>) -> PyResult<(Vec<u8>, Vec<usize>)> {
    let shape = arr.shape().to_vec();
    if !(shape.len() == 2 || (shape.len() == 3 && (shape[2] == 1 || shape[2] == 3))) {
        return Err(PyValueError::new_err("expected 2D (H,W) or 3D (H,W,1|3) uint8 array"));
    }
    let flat = arr
        .as_array()
        .to_owned()
        .into_raw_vec();
    Ok((flat, shape))
}

fn py_u8_ndarray_to_vec_and_shape(arr: &PyReadonlyArrayDyn<u8>) -> PyResult<(Vec<u8>, Vec<usize>)> {
    let shape = arr.shape().to_vec();
    if shape.is_empty() {
        return Err(PyValueError::new_err("expected a non-scalar uint8 array"));
    }
    let flat = arr.as_array().to_owned().into_raw_vec();
    Ok((flat, shape))
}

fn ensure_image_3ch(shape: &[usize]) -> PyResult<(usize, usize)> {
    if shape.len() != 3 || shape[2] != 3 {
        return Err(PyValueError::new_err("expected (H,W,3) uint8 array"));
    }
    Ok((shape[0], shape[1]))
}

fn ensure_same_shape(a: &[usize], b: &[usize]) -> PyResult<()> {
    if a != b {
        return Err(PyValueError::new_err("inputs must have the same shape"));
    }
    Ok(())
}

fn parse_optional_dst_u8<'py>(obj: Option<&Bound<'py, PyAny>>) -> PyResult<Option<Bound<'py, PyArrayDyn<u8>>>> {
    match obj {
        None => Ok(None),
        Some(o) if o.is_none() => Ok(None),
        Some(o) => Ok(Some(
            o.downcast::<PyArrayDyn<u8>>()
                .map_err(|_| PyTypeError::new_err("dst must be a uint8 numpy array"))?
                .to_owned(),
        )),
    }
}

fn extract_u8_array<'py>(obj: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyArrayDyn<u8>>> {
    obj.downcast::<PyArrayDyn<u8>>()
        .map(|a| a.to_owned())
        .map_err(|_| PyTypeError::new_err("expected a uint8 numpy array"))
}

fn write_or_return_u8<'py>(
    py: Python<'py>,
    out: Vec<u8>,
    shape: &[usize],
    dst: Option<Bound<'py, PyArrayDyn<u8>>>,
) -> PyResult<Py<PyArrayDyn<u8>>> {
    if let Some(dst) = dst {
        let mut dst_rw = dst
            .try_readwrite()
            .map_err(|_| PyValueError::new_err("dst is not writable"))?;
        write_vec_to_pyarray(&mut dst_rw, &out, shape)?;
        Ok(dst.unbind())
    } else {
        pyarray_from_vec(py, out, shape)
    }
}

fn extract_optional_mask_u8(mask: Option<PyReadonlyArrayDyn<'_, u8>>, shape: &[usize]) -> PyResult<Option<Vec<u8>>> {
    let Some(mask) = mask else {
        return Ok(None);
    };
    let mask_shape = mask.shape().to_vec();
    let (mh, mw) = match mask_shape.as_slice() {
        [mh, mw] => (*mh, *mw),
        [mh, mw, 1] => (*mh, *mw),
        _ => return Err(PyValueError::new_err("expected mask shape (H,W) or (H,W,1)")),
    };

    let (h, w) = match shape {
        [h, w] => (*h, *w),
        [h, w, _c] => (*h, *w),
        _ => return Err(PyValueError::new_err("unsupported dims")),
    };
    if mh != h || mw != w {
        return Err(PyValueError::new_err("mask must match image height/width"));
    }

    Ok(Some(mask.as_array().to_owned().into_raw_vec()))
}

fn apply_mask_u8(out: &mut [u8], base: &[u8], shape: &[usize], mask: &[u8]) -> PyResult<()> {
    if out.len() != base.len() {
        return Err(PyValueError::new_err("internal error: base/out size mismatch"));
    }
    let channels = match shape {
        [_h, _w] => 1usize,
        [_h, _w, c] => *c,
        _ => return Err(PyValueError::new_err("unsupported dims")),
    };
    if !(channels == 1 || channels == 3) {
        return Err(PyValueError::new_err("expected 1 or 3 channels"));
    }
    let pixels = mask.len();
    if pixels * channels != out.len() {
        return Err(PyValueError::new_err("internal error: mask/shape mismatch"));
    }
    for i in 0..pixels {
        if mask[i] == 0 {
            let base_idx = i * channels;
            for c in 0..channels {
                out[base_idx + c] = base[base_idx + c];
            }
        }
    }
    Ok(())
}

fn clamp_i32(v: i32, lo: i32, hi: i32) -> i32 {
    if v < lo {
        lo
    } else if v > hi {
        hi
    } else {
        v
    }
}

fn convolve_separable_u8(
    src: &[u8],
    h: usize,
    w: usize,
    c: usize,
    kx: &[f32],
    ky: &[f32],
) -> Vec<u8> {
    let rx = (kx.len() / 2) as i32;
    let ry = (ky.len() / 2) as i32;
    let mut tmp = vec![0f32; h * w * c];

    // Horizontal
    for y in 0..h {
        for x in 0..w {
            for ch in 0..c {
                let mut acc = 0f32;
                for (i, &kw) in kx.iter().enumerate() {
                    let xx = clamp_i32(x as i32 + i as i32 - rx, 0, (w - 1) as i32) as usize;
                    let idx = (y * w + xx) * c + ch;
                    acc += kw * (src[idx] as f32);
                }
                tmp[(y * w + x) * c + ch] = acc;
            }
        }
    }

    // Vertical
    let mut out = vec![0u8; h * w * c];
    for y in 0..h {
        for x in 0..w {
            for ch in 0..c {
                let mut acc = 0f32;
                for (j, &kw) in ky.iter().enumerate() {
                    let yy = clamp_i32(y as i32 + j as i32 - ry, 0, (h - 1) as i32) as usize;
                    acc += kw * tmp[(yy * w + x) * c + ch];
                }
                out[(y * w + x) * c + ch] = acc.round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    out
}

fn convolve_separable_i32(
    src: &[u8],
    h: usize,
    w: usize,
    c: usize,
    kx: &[i32],
    ky: &[i32],
) -> Vec<i32> {
    let rx = (kx.len() / 2) as i32;
    let ry = (ky.len() / 2) as i32;
    let mut tmp = vec![0i32; h * w * c];

    for y in 0..h {
        for x in 0..w {
            for ch in 0..c {
                let mut acc = 0i32;
                for (i, &kw) in kx.iter().enumerate() {
                    let xx = clamp_i32(x as i32 + i as i32 - rx, 0, (w - 1) as i32) as usize;
                    let idx = (y * w + xx) * c + ch;
                    acc += kw * (src[idx] as i32);
                }
                tmp[(y * w + x) * c + ch] = acc;
            }
        }
    }

    let mut out = vec![0i32; h * w * c];
    for y in 0..h {
        for x in 0..w {
            for ch in 0..c {
                let mut acc = 0i32;
                for (j, &kw) in ky.iter().enumerate() {
                    let yy = clamp_i32(y as i32 + j as i32 - ry, 0, (h - 1) as i32) as usize;
                    acc += kw * tmp[(yy * w + x) * c + ch];
                }
                out[(y * w + x) * c + ch] = acc;
            }
        }
    }
    out
}

fn gaussian_kernel_1d(ksize: usize, sigma: f64) -> PyResult<Vec<f32>> {
    if ksize == 0 || ksize % 2 == 0 {
        return Err(PyValueError::new_err("ksize must be positive odd"));
    }
    if !sigma.is_finite() || sigma < 0.0 {
        return Err(PyValueError::new_err("sigma must be finite and >= 0"));
    }
    // OpenCV: if sigma == 0, it estimates based on ksize.
    let sigma = if sigma == 0.0 {
        // Roughly matches OpenCV's internal heuristic.
        // (ksize-1)*0.5 -> radius
        let r = (ksize as f64 - 1.0) * 0.5;
        0.3 * (r - 1.0) + 0.8
    } else {
        sigma
    };
    let mut k = vec![0f64; ksize];
    let r = (ksize / 2) as i32;
    let denom = 2.0 * sigma * sigma;
    for i in 0..ksize {
        let x = (i as i32 - r) as f64;
        k[i] = (-x * x / denom).exp();
    }
    let sum: f64 = k.iter().sum();
    let out = k.into_iter().map(|v| (v / sum) as f32).collect();
    Ok(out)
}

fn box_kernel_1d(ksize: usize) -> PyResult<Vec<f32>> {
    if ksize == 0 {
        return Err(PyValueError::new_err("ksize must be positive"));
    }
    let v = 1.0f32 / (ksize as f32);
    Ok(vec![v; ksize])
}

fn parse_image_shape(shape: &[usize]) -> PyResult<(usize, usize, usize)> {
    match shape {
        [h, w] => Ok((*h, *w, 1)),
        [h, w, c] if *c == 1 || *c == 3 => Ok((*h, *w, *c)),
        _ => Err(PyValueError::new_err(
            "expected 2D (H,W) or 3D (H,W,1|3) uint8 array",
        )),
    }
}

#[pyfunction]
#[pyo3(signature = (src, flipCode))]
fn flip(py: Python<'_>, src: PyReadonlyArrayDyn<u8>, flipCode: i32) -> PyResult<Py<PyArrayDyn<u8>>> {
    let (buf, shape) = py_image_ndarray_to_vec_and_shape(&src)?;
    let (h, w, c) = parse_image_shape(&shape)?;
    let mut out = vec![0u8; buf.len()];

    // OpenCV: flipCode == 0 => x-axis (vertical), >0 => y-axis (horizontal), <0 => both.
    for y in 0..h {
        for x in 0..w {
            let (yy, xx) = if flipCode == 0 {
                (h - 1 - y, x)
            } else if flipCode > 0 {
                (y, w - 1 - x)
            } else {
                (h - 1 - y, w - 1 - x)
            };
            let src_idx = (y * w + x) * c;
            let dst_idx = (yy * w + xx) * c;
            out[dst_idx..dst_idx + c].copy_from_slice(&buf[src_idx..src_idx + c]);
        }
    }
    pyarray_from_vec(py, out, &shape)
}

#[pyfunction]
fn transpose(py: Python<'_>, src: PyReadonlyArrayDyn<u8>) -> PyResult<Py<PyArrayDyn<u8>>> {
    let (buf, shape) = py_image_ndarray_to_vec_and_shape(&src)?;
    let (h, w, c) = parse_image_shape(&shape)?;
    let mut out = vec![0u8; buf.len()];
    for y in 0..h {
        for x in 0..w {
            let src_idx = (y * w + x) * c;
            let dst_idx = (x * h + y) * c;
            out[dst_idx..dst_idx + c].copy_from_slice(&buf[src_idx..src_idx + c]);
        }
    }
    let out_shape = if c == 1 {
        vec![w, h]
    } else {
        vec![w, h, c]
    };
    pyarray_from_vec(py, out, &out_shape)
}

#[pyfunction]
#[pyo3(signature = (src, rotateCode))]
fn rotate(py: Python<'_>, src: PyReadonlyArrayDyn<u8>, rotateCode: i32) -> PyResult<Py<PyArrayDyn<u8>>> {
    let (buf, shape) = py_image_ndarray_to_vec_and_shape(&src)?;
    let (h, w, c) = parse_image_shape(&shape)?;

    match rotateCode {
        ROTATE_180 => {
            // 180: flip both axes
            let mut out = vec![0u8; buf.len()];
            for y in 0..h {
                for x in 0..w {
                    let src_idx = (y * w + x) * c;
                    let dst_idx = ((h - 1 - y) * w + (w - 1 - x)) * c;
                    out[dst_idx..dst_idx + c].copy_from_slice(&buf[src_idx..src_idx + c]);
                }
            }
            pyarray_from_vec(py, out, &shape)
        }
        ROTATE_90_CLOCKWISE => {
            // dst shape: (w,h)
            let mut out = vec![0u8; buf.len()];
            for y in 0..h {
                for x in 0..w {
                    let src_idx = (y * w + x) * c;
                    let dst_y = x;
                    let dst_x = h - 1 - y;
                    let dst_idx = (dst_y * h + dst_x) * c;
                    out[dst_idx..dst_idx + c].copy_from_slice(&buf[src_idx..src_idx + c]);
                }
            }
            let out_shape = if c == 1 { vec![w, h] } else { vec![w, h, c] };
            pyarray_from_vec(py, out, &out_shape)
        }
        ROTATE_90_COUNTERCLOCKWISE => {
            let mut out = vec![0u8; buf.len()];
            for y in 0..h {
                for x in 0..w {
                    let src_idx = (y * w + x) * c;
                    let dst_y = w - 1 - x;
                    let dst_x = y;
                    let dst_idx = (dst_y * h + dst_x) * c;
                    out[dst_idx..dst_idx + c].copy_from_slice(&buf[src_idx..src_idx + c]);
                }
            }
            let out_shape = if c == 1 { vec![w, h] } else { vec![w, h, c] };
            pyarray_from_vec(py, out, &out_shape)
        }
        _ => Err(PyValueError::new_err("unsupported rotate code")),
    }
}

fn write_vec_to_pyarray(dst: &mut PyReadwriteArrayDyn<u8>, data: &[u8], shape: &[usize]) -> PyResult<()> {
    if dst.shape() != shape {
        return Err(PyValueError::new_err(format!(
            "dst has shape {:?} but expected {:?}",
            dst.shape(),
            shape
        )));
    }

    let src_view = ArrayViewD::from_shape(IxDyn(shape), data)
        .map_err(|e| PyValueError::new_err(format!("invalid shape: {e}")))?;
    dst.as_array_mut().assign(&src_view);
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (filename, *args))]
fn imread(py: Python<'_>, filename: PathBuf, args: &Bound<'_, pyo3::types::PyTuple>) -> PyResult<Option<Py<PyArrayDyn<u8>>>> {
    // OpenCV returns None on I/O/decode failures; match that behavior.
    //
    // Upstream OpenCV's Python bindings also expose an overload that writes into a
    // preallocated destination array: `imread(filename, dst[, flags])`.
    // We support both:
    //   - imread(filename[, flags])
    //   - imread(filename, dst[, flags])

    let mut dst: Option<Bound<'_, PyArrayDyn<u8>>> = None;
    let mut flags: i32 = IMREAD_COLOR;

    match args.len() {
        0 => {}
        1 => {
            let a0 = args.get_item(0)?;
            if let Ok(v) = a0.extract::<i32>() {
                flags = v;
            } else if let Ok(arr) = a0.downcast::<PyArrayDyn<u8>>() {
                dst = Some(arr.to_owned());
            } else {
                return Err(PyTypeError::new_err(
                    "imread() second argument must be flags (int) or dst (np.ndarray[uint8])",
                ));
            }
        }
        2 => {
            let a0 = args.get_item(0)?;
            let a1 = args.get_item(1)?;
            dst = Some(
                a0.downcast::<PyArrayDyn<u8>>()
                    .map_err(|_| PyTypeError::new_err("imread() dst must be a uint8 numpy array"))?
                    .to_owned(),
            );
            flags = a1
                .extract::<i32>()
                .map_err(|_| PyTypeError::new_err("imread() flags must be an int"))?;
        }
        _ => {
            return Err(PyTypeError::new_err(
                "imread() takes (filename[, flags]) or (filename, dst[, flags])",
            ))
        }
    }

    let bytes = match std::fs::read(&filename) {
        Ok(b) => b,
        Err(_) => return Ok(None),
    };
    let decoded = match decode_image_bytes(py, &bytes, flags) {
        Ok(img) => img,
        Err(_) => return Ok(None),
    };

    if let Some(dst) = dst {
        let decoded_ro = decoded.bind(py).readonly();
        let (buf, shape) = py_image_ndarray_to_vec_and_shape(&decoded_ro)?;
        let mut dst_rw = dst
            .try_readwrite()
            .map_err(|_| PyValueError::new_err("imread() dst is not writable"))?;
        write_vec_to_pyarray(&mut dst_rw, &buf, &shape)?;
        // Match upstream behavior: when dst is provided, return dst.
        return Ok(Some(dst.unbind()));
    }

    Ok(Some(decoded))
}

#[pyfunction]
fn imwrite(path: PathBuf, img: PyReadonlyArrayDyn<u8>) -> PyResult<bool> {
    let (buf, shape) = py_image_ndarray_to_vec_and_shape(&img)?;
    let save_res = match shape.len() {
        2 => {
            let h = shape[0] as u32;
            let w = shape[1] as u32;
            let out = image::GrayImage::from_raw(w, h, buf)
                .ok_or_else(|| PyValueError::new_err("invalid grayscale image shape"))?;
            out.save(&path)
        }
        3 => {
            let h = shape[0] as u32;
            let w = shape[1] as u32;
            let c = shape[2];
            if c == 1 {
                let out = image::GrayImage::from_raw(w, h, buf)
                    .ok_or_else(|| PyValueError::new_err("invalid grayscale image shape"))?;
                out.save(&path)
            } else if c == 3 {
                // Convert BGR -> RGB for saving
                let mut rgb = buf;
                rgb_to_bgr_inplace(&mut rgb);
                let out = image::RgbImage::from_raw(w, h, rgb)
                    .ok_or_else(|| PyValueError::new_err("invalid color image shape"))?;
                out.save(&path)
            } else {
                return Err(PyValueError::new_err("expected 1 or 3 channels"));
            }
        }
        _ => return Err(PyValueError::new_err("unexpected image dims")),
    };

    // OpenCV returns a boolean; treat I/O errors as a simple failure.
    Ok(save_res.is_ok())
}

#[pyfunction]
#[pyo3(signature = (buf, flags=IMREAD_COLOR))]
fn imdecode(py: Python<'_>, buf: &Bound<'_, PyAny>, flags: i32) -> PyResult<Option<Py<PyArrayDyn<u8>>>> {
    let bytes = any_to_bytes(buf)?;
    match decode_image_bytes(py, &bytes, flags) {
        Ok(img) => Ok(Some(img)),
        Err(_) => Ok(None),
    }
}

#[pyfunction]
#[pyo3(signature = (ext, img))]
fn imencode(py: Python<'_>, ext: &str, img: PyReadonlyArrayDyn<u8>) -> PyResult<(bool, Py<PyArrayDyn<u8>>)> {
    let (buf, shape) = py_image_ndarray_to_vec_and_shape(&img)?;

    let out_format = match ext.to_ascii_lowercase().as_str() {
        ".png" => image::ImageOutputFormat::Png,
        ".jpg" | ".jpeg" => image::ImageOutputFormat::Jpeg(90),
        _ => return Err(PyValueError::new_err("unsupported format; use .png/.jpg/.jpeg")),
    };

    let dyn_img = match shape.len() {
        2 => {
            let h = shape[0] as u32;
            let w = shape[1] as u32;
            let gray = image::GrayImage::from_raw(w, h, buf)
                .ok_or_else(|| PyValueError::new_err("invalid grayscale image shape"))?;
            DynamicImage::ImageLuma8(gray)
        }
        3 => {
            let h = shape[0] as u32;
            let w = shape[1] as u32;
            let c = shape[2];
            if c == 1 {
                let gray = image::GrayImage::from_raw(w, h, buf)
                    .ok_or_else(|| PyValueError::new_err("invalid grayscale image shape"))?;
                DynamicImage::ImageLuma8(gray)
            } else if c == 3 {
                // Convert BGR -> RGB for encoding.
                let mut rgb = buf;
                rgb_to_bgr_inplace(&mut rgb);
                let rgb_img = image::RgbImage::from_raw(w, h, rgb)
                    .ok_or_else(|| PyValueError::new_err("invalid color image shape"))?;
                DynamicImage::ImageRgb8(rgb_img)
            } else {
                return Err(PyValueError::new_err("expected 1 or 3 channels"));
            }
        }
        _ => return Err(PyValueError::new_err("unexpected image dims")),
    };

    let mut cursor = Cursor::new(Vec::<u8>::new());
    if dyn_img.write_to(&mut cursor, out_format).is_err() {
        return Ok((false, pyarray_from_vec(py, vec![], &[0])?));
    }
    let out_bytes = cursor.into_inner();
    let out_len = out_bytes.len();
    let out_arr = pyarray_from_vec(py, out_bytes, &[out_len])?;
    Ok((true, out_arr))
}

#[pyfunction]
#[pyo3(signature = (*args))]
fn norm(py: Python<'_>, args: &Bound<'_, pyo3::types::PyTuple>) -> PyResult<f64> {
    // Minimal OpenCV-like `norm` support for uint8 images.
    // Supported:
    //   - norm(src1)
    //   - norm(src1, normType)
    //   - norm(src1, src2)
    //   - norm(src1, src2, normType)
    let (src1_obj, src2_obj, norm_type) = match args.len() {
        1 => (args.get_item(0)?, None, NORM_L2),
        2 => {
            let a0 = args.get_item(0)?;
            let a1 = args.get_item(1)?;
            if a1.extract::<i32>().is_ok() {
                (a0, None, a1.extract::<i32>()?)
            } else {
                (a0, Some(a1), NORM_L2)
            }
        }
        3 => (
            args.get_item(0)?,
            Some(args.get_item(1)?),
            args.get_item(2)?.extract::<i32>()?,
        ),
        _ => {
            return Err(PyTypeError::new_err(
                "norm() takes (src1), (src1, normType), (src1, src2), or (src1, src2, normType)",
            ))
        }
    };

    let src1 = src1_obj
        .extract::<PyReadonlyArrayDyn<'_, u8>>()
        .map_err(|_| PyTypeError::new_err("norm() expects uint8 numpy arrays"))?;
    let (a_buf, a_shape) = py_u8_ndarray_to_vec_and_shape(&src1)?;

    let b_buf = if let Some(src2_obj) = src2_obj {
        let src2 = src2_obj
            .extract::<PyReadonlyArrayDyn<'_, u8>>()
            .map_err(|_| PyTypeError::new_err("norm() expects uint8 numpy arrays"))?;
        let (b_buf, b_shape) = py_u8_ndarray_to_vec_and_shape(&src2)?;
        if b_shape != a_shape {
            return Err(PyValueError::new_err("src1 and src2 must have the same shape"));
        }
        Some(b_buf)
    } else {
        None
    };

    let result = match norm_type {
        NORM_INF => {
            let mut m: f64 = 0.0;
            if let Some(b_iter) = b_buf.as_ref() {
                for (a, b) in a_buf.iter().zip(b_iter.iter()) {
                    let d = (*a as i16 - *b as i16).abs() as f64;
                    if d > m {
                        m = d;
                    }
                }
            } else {
                for &a in a_buf.iter() {
                    let d = a as f64;
                    if d > m {
                        m = d;
                    }
                }
            }
            m
        }
        NORM_L1 => {
            let mut s: f64 = 0.0;
            if let Some(b) = b_buf.as_ref() {
                for (a, b) in a_buf.iter().zip(b.iter()) {
                    s += (*a as i16 - *b as i16).abs() as f64;
                }
            } else {
                for &a in a_buf.iter() {
                    s += a as f64;
                }
            }
            s
        }
        NORM_L2 => {
            let mut s: f64 = 0.0;
            if let Some(b) = b_buf.as_ref() {
                for (a, b) in a_buf.iter().zip(b.iter()) {
                    let d = (*a as f64) - (*b as f64);
                    s += d * d;
                }
            } else {
                for &a in a_buf.iter() {
                    let d = a as f64;
                    s += d * d;
                }
            }
            s.sqrt()
        }
        _ => return Err(PyValueError::new_err("unsupported norm type")),
    };

    // Keep `py` in signature to follow PyO3 patterns and avoid surprising GIL releases.
    let _ = py;
    Ok(result)
}

#[pyfunction]
fn absdiff(py: Python<'_>, src1: PyReadonlyArrayDyn<u8>, src2: PyReadonlyArrayDyn<u8>) -> PyResult<Py<PyArrayDyn<u8>>> {
    let (a_buf, a_shape) = py_u8_ndarray_to_vec_and_shape(&src1)?;
    let (b_buf, b_shape) = py_u8_ndarray_to_vec_and_shape(&src2)?;
    if a_shape != b_shape {
        return Err(PyValueError::new_err("src1 and src2 must have the same shape"));
    }

    let out: Vec<u8> = a_buf
        .iter()
        .zip(b_buf.iter())
        .map(|(a, b)| (*a as i16 - *b as i16).abs().clamp(0, 255) as u8)
        .collect();
    pyarray_from_vec(py, out, &a_shape)
}

#[pyfunction]
fn add(py: Python<'_>, src1: PyReadonlyArrayDyn<u8>, src2: PyReadonlyArrayDyn<u8>) -> PyResult<Py<PyArrayDyn<u8>>> {
    let (a_buf, a_shape) = py_u8_ndarray_to_vec_and_shape(&src1)?;
    let (b_buf, b_shape) = py_u8_ndarray_to_vec_and_shape(&src2)?;
    if a_shape != b_shape {
        return Err(PyValueError::new_err("src1 and src2 must have the same shape"));
    }

    let out: Vec<u8> = a_buf
        .iter()
        .zip(b_buf.iter())
        .map(|(a, b)| ((*a as i16 + *b as i16).clamp(0, 255)) as u8)
        .collect();
    pyarray_from_vec(py, out, &a_shape)
}

#[pyfunction]
fn subtract(py: Python<'_>, src1: PyReadonlyArrayDyn<u8>, src2: PyReadonlyArrayDyn<u8>) -> PyResult<Py<PyArrayDyn<u8>>> {
    let (a_buf, a_shape) = py_u8_ndarray_to_vec_and_shape(&src1)?;
    let (b_buf, b_shape) = py_u8_ndarray_to_vec_and_shape(&src2)?;
    if a_shape != b_shape {
        return Err(PyValueError::new_err("src1 and src2 must have the same shape"));
    }

    let out: Vec<u8> = a_buf
        .iter()
        .zip(b_buf.iter())
        .map(|(a, b)| ((*a as i16 - *b as i16).clamp(0, 255)) as u8)
        .collect();
    pyarray_from_vec(py, out, &a_shape)
}

#[pyfunction]
fn split(py: Python<'_>, src: PyReadonlyArrayDyn<u8>) -> PyResult<Py<PyList>> {
    let (buf, shape) = py_image_ndarray_to_vec_and_shape(&src)?;
    let (h, w) = ensure_image_3ch(&shape)?;

    let mut c0 = Vec::with_capacity(h * w);
    let mut c1 = Vec::with_capacity(h * w);
    let mut c2 = Vec::with_capacity(h * w);
    for px in buf.chunks_exact(3) {
        c0.push(px[0]);
        c1.push(px[1]);
        c2.push(px[2]);
    }

    let out0 = pyarray_from_vec(py, c0, &[h, w])?;
    let out1 = pyarray_from_vec(py, c1, &[h, w])?;
    let out2 = pyarray_from_vec(py, c2, &[h, w])?;

    let list = PyList::empty_bound(py);
    list.append(out0)?;
    list.append(out1)?;
    list.append(out2)?;
    Ok(list.unbind())
}

#[pyfunction]
fn merge(py: Python<'_>, seq: &Bound<'_, PyAny>) -> PyResult<Py<PyArrayDyn<u8>>> {
    // Minimal `cv.merge([c0, c1, c2])` implementation for uint8.
    let seq = seq
        .downcast::<PySequence>()
        .map_err(|_| PyTypeError::new_err("merge() expects a sequence of arrays"))?;
    if seq.len()? != 3 {
        return Err(PyValueError::new_err("merge() expects exactly 3 channels"));
    }

    let c0 = seq.get_item(0)?.extract::<PyReadonlyArrayDyn<'_, u8>>()?;
    let c1 = seq.get_item(1)?.extract::<PyReadonlyArrayDyn<'_, u8>>()?;
    let c2 = seq.get_item(2)?.extract::<PyReadonlyArrayDyn<'_, u8>>()?;

    let (b0, s0) = py_u8_ndarray_to_vec_and_shape(&c0)?;
    let (b1, s1) = py_u8_ndarray_to_vec_and_shape(&c1)?;
    let (b2, s2) = py_u8_ndarray_to_vec_and_shape(&c2)?;

    if s0.len() != 2 || s1.len() != 2 || s2.len() != 2 {
        return Err(PyValueError::new_err("merge() expects 2D channel arrays"));
    }
    if s0 != s1 || s0 != s2 {
        return Err(PyValueError::new_err("all channels must have the same shape"));
    }
    let h = s0[0];
    let w = s0[1];
    if b0.len() != h * w || b1.len() != h * w || b2.len() != h * w {
        return Err(PyValueError::new_err("invalid channel buffer length"));
    }

    let mut out = Vec::with_capacity(h * w * 3);
    for i in 0..(h * w) {
        out.push(b0[i]);
        out.push(b1[i]);
        out.push(b2[i]);
    }
    pyarray_from_vec(py, out, &[h, w, 3])
}

#[pyfunction]
#[pyo3(signature = (src, dst=None, mask=None))]
fn bitwise_not(
    py: Python<'_>,
    src: &Bound<'_, PyAny>,
    dst: Option<&Bound<'_, PyAny>>,
    mask: Option<PyReadonlyArrayDyn<'_, u8>>,
) -> PyResult<Py<PyArrayDyn<u8>>> {
    let src_arr = extract_u8_array(src)?;
    let shape = src_arr.shape().to_vec();
    let dst_arr = parse_optional_dst_u8(dst)?;
    let mask = extract_optional_mask_u8(mask, &shape)?;

    let src_buf = {
        let ro = src_arr.readonly();
        ro.as_array().to_owned().into_raw_vec()
    };
    let mut out: Vec<u8> = src_buf.iter().map(|&v| 255u8.wrapping_sub(v)).collect();

    if let Some(mask) = mask.as_ref() {
        let base = if let Some(ref d) = dst_arr {
            ensure_same_shape(d.shape(), &shape)?;
            d.readonly().as_array().to_owned().into_raw_vec()
        } else {
            vec![0u8; out.len()]
        };
        apply_mask_u8(&mut out, &base, &shape, mask)?;
    }
    if let Some(ref d) = dst_arr {
        ensure_same_shape(d.shape(), &shape)?;
    }
    write_or_return_u8(py, out, &shape, dst_arr)
}

fn bitwise_binop<F>(a: &[u8], b: &[u8], f: F) -> Vec<u8>
where
    F: Fn(u8, u8) -> u8,
{
    a.iter().zip(b.iter()).map(|(&x, &y)| f(x, y)).collect()
}

#[pyfunction]
#[pyo3(signature = (src1, src2, dst=None, mask=None))]
fn bitwise_and(
    py: Python<'_>,
    src1: &Bound<'_, PyAny>,
    src2: &Bound<'_, PyAny>,
    dst: Option<&Bound<'_, PyAny>>,
    mask: Option<PyReadonlyArrayDyn<'_, u8>>,
) -> PyResult<Py<PyArrayDyn<u8>>> {
    let src1_arr = extract_u8_array(src1)?;
    let src2_arr = extract_u8_array(src2)?;
    let shape_a = src1_arr.shape().to_vec();
    let shape_b = src2_arr.shape().to_vec();
    ensure_same_shape(&shape_a, &shape_b)?;

    let a = {
        let ro = src1_arr.readonly();
        ro.as_array().to_owned().into_raw_vec()
    };
    let b = {
        let ro = src2_arr.readonly();
        ro.as_array().to_owned().into_raw_vec()
    };

    let dst = parse_optional_dst_u8(dst)?;
    let mask = extract_optional_mask_u8(mask, &shape_a)?;
    if let Some(ref d) = dst {
        ensure_same_shape(d.shape(), &shape_a)?;
    }

    let mut out = bitwise_binop(&a, &b, |x, y| x & y);
    if let Some(mask) = mask.as_ref() {
        let base = if let Some(ref d) = dst {
            d.readonly().as_array().to_owned().into_raw_vec()
        } else {
            vec![0u8; out.len()]
        };
        apply_mask_u8(&mut out, &base, &shape_a, mask)?;
    }
    write_or_return_u8(py, out, &shape_a, dst)
}

#[pyfunction]
#[pyo3(signature = (src1, src2, dst=None, mask=None))]
fn bitwise_or(
    py: Python<'_>,
    src1: &Bound<'_, PyAny>,
    src2: &Bound<'_, PyAny>,
    dst: Option<&Bound<'_, PyAny>>,
    mask: Option<PyReadonlyArrayDyn<'_, u8>>,
) -> PyResult<Py<PyArrayDyn<u8>>> {
    let src1_arr = extract_u8_array(src1)?;
    let src2_arr = extract_u8_array(src2)?;
    let shape_a = src1_arr.shape().to_vec();
    let shape_b = src2_arr.shape().to_vec();
    ensure_same_shape(&shape_a, &shape_b)?;

    let a = {
        let ro = src1_arr.readonly();
        ro.as_array().to_owned().into_raw_vec()
    };
    let b = {
        let ro = src2_arr.readonly();
        ro.as_array().to_owned().into_raw_vec()
    };

    let dst = parse_optional_dst_u8(dst)?;
    let mask = extract_optional_mask_u8(mask, &shape_a)?;
    if let Some(ref d) = dst {
        ensure_same_shape(d.shape(), &shape_a)?;
    }

    let mut out = bitwise_binop(&a, &b, |x, y| x | y);
    if let Some(mask) = mask.as_ref() {
        let base = if let Some(ref d) = dst {
            d.readonly().as_array().to_owned().into_raw_vec()
        } else {
            vec![0u8; out.len()]
        };
        apply_mask_u8(&mut out, &base, &shape_a, mask)?;
    }
    write_or_return_u8(py, out, &shape_a, dst)
}

#[pyfunction]
#[pyo3(signature = (src1, src2, dst=None, mask=None))]
fn bitwise_xor(
    py: Python<'_>,
    src1: &Bound<'_, PyAny>,
    src2: &Bound<'_, PyAny>,
    dst: Option<&Bound<'_, PyAny>>,
    mask: Option<PyReadonlyArrayDyn<'_, u8>>,
) -> PyResult<Py<PyArrayDyn<u8>>> {
    let src1_arr = extract_u8_array(src1)?;
    let src2_arr = extract_u8_array(src2)?;
    let shape_a = src1_arr.shape().to_vec();
    let shape_b = src2_arr.shape().to_vec();
    ensure_same_shape(&shape_a, &shape_b)?;

    let a = {
        let ro = src1_arr.readonly();
        ro.as_array().to_owned().into_raw_vec()
    };
    let b = {
        let ro = src2_arr.readonly();
        ro.as_array().to_owned().into_raw_vec()
    };

    let dst = parse_optional_dst_u8(dst)?;
    let mask = extract_optional_mask_u8(mask, &shape_a)?;
    if let Some(ref d) = dst {
        ensure_same_shape(d.shape(), &shape_a)?;
    }

    let mut out = bitwise_binop(&a, &b, |x, y| x ^ y);
    if let Some(mask) = mask.as_ref() {
        let base = if let Some(ref d) = dst {
            d.readonly().as_array().to_owned().into_raw_vec()
        } else {
            vec![0u8; out.len()]
        };
        apply_mask_u8(&mut out, &base, &shape_a, mask)?;
    }
    write_or_return_u8(py, out, &shape_a, dst)
}

#[pyfunction]
#[pyo3(signature = (src, thresh, maxval, typ))]
fn threshold(
    py: Python<'_>,
    src: PyReadonlyArrayDyn<u8>,
    thresh: f64,
    maxval: f64,
    typ: i32,
) -> PyResult<(f64, Py<PyArrayDyn<u8>>)> {
    // Minimal threshold for uint8 arrays.
    // OpenCV accepts various dtypes and has OTSU/TRIANGLE, but we keep it small.
    let (buf, shape) = py_u8_ndarray_to_vec_and_shape(&src)?;
    if thresh.is_nan() || maxval.is_nan() {
        return Err(PyValueError::new_err("thresh/maxval must be finite"));
    }
    let t = thresh.clamp(0.0, 255.0) as u8;
    let mv = maxval.clamp(0.0, 255.0) as u8;

    let out: Vec<u8> = match typ {
        THRESH_BINARY => buf.iter().map(|&v| if v > t { mv } else { 0 }).collect(),
        THRESH_BINARY_INV => buf.iter().map(|&v| if v > t { 0 } else { mv }).collect(),
        THRESH_TRUNC => buf.iter().map(|&v| if v > t { t } else { v }).collect(),
        THRESH_TOZERO => buf.iter().map(|&v| if v > t { v } else { 0 }).collect(),
        THRESH_TOZERO_INV => buf.iter().map(|&v| if v > t { 0 } else { v }).collect(),
        _ => return Err(PyValueError::new_err("unsupported threshold type")),
    };

    Ok((thresh, pyarray_from_vec(py, out, &shape)?))
}

#[pyfunction]
fn countNonZero(src: PyReadonlyArrayDyn<u8>) -> PyResult<i32> {
    let (buf, _shape) = py_u8_ndarray_to_vec_and_shape(&src)?;
    let count = buf.iter().filter(|&&v| v != 0).count();
    Ok(count
        .try_into()
        .map_err(|_| PyValueError::new_err("count too large"))?)
}

#[pyfunction]
#[pyo3(signature = (src1, src2, cmpop))]
fn compare(
    py: Python<'_>,
    src1: PyReadonlyArrayDyn<u8>,
    src2: PyReadonlyArrayDyn<u8>,
    cmpop: i32,
) -> PyResult<Py<PyArrayDyn<u8>>> {
    let (a, shape_a) = py_u8_ndarray_to_vec_and_shape(&src1)?;
    let (b, shape_b) = py_u8_ndarray_to_vec_and_shape(&src2)?;
    ensure_same_shape(&shape_a, &shape_b)?;

    let out: Vec<u8> = match cmpop {
        CMP_EQ => bitwise_binop(&a, &b, |x, y| if x == y { 255 } else { 0 }),
        CMP_NE => bitwise_binop(&a, &b, |x, y| if x != y { 255 } else { 0 }),
        CMP_LT => bitwise_binop(&a, &b, |x, y| if x < y { 255 } else { 0 }),
        CMP_LE => bitwise_binop(&a, &b, |x, y| if x <= y { 255 } else { 0 }),
        CMP_GT => bitwise_binop(&a, &b, |x, y| if x > y { 255 } else { 0 }),
        CMP_GE => bitwise_binop(&a, &b, |x, y| if x >= y { 255 } else { 0 }),
        _ => return Err(PyValueError::new_err("unsupported compare op")),
    };
    pyarray_from_vec(py, out, &shape_a)
}

#[pyfunction]
fn inRange(
    py: Python<'_>,
    src: PyReadonlyArrayDyn<u8>,
    lowerb: &Bound<'_, PyAny>,
    upperb: &Bound<'_, PyAny>,
) -> PyResult<Py<PyArrayDyn<u8>>> {
    // Minimal `inRange` for uint8 images.
    // - For grayscale: src shape (H,W), bounds are scalars.
    // - For color: src shape (H,W,3), bounds are scalars or 3-element sequences.
    let (buf, shape) = py_image_ndarray_to_vec_and_shape(&src)?;

    let (h, w, channels) = match shape.len() {
        2 => (shape[0], shape[1], 1usize),
        3 => (shape[0], shape[1], shape[2]),
        _ => return Err(PyValueError::new_err("unsupported dims")),
    };
    if !(channels == 1 || channels == 3) {
        return Err(PyValueError::new_err("expected 1 or 3 channels"));
    }

    let lo = vec_u8_from_scalar_or_seq(lowerb, channels)?;
    let hi = vec_u8_from_scalar_or_seq(upperb, channels)?;

    let mut out = Vec::with_capacity(h * w);
    if channels == 1 {
        let lo0 = lo[0];
        let hi0 = hi[0];
        for &v in buf.iter() {
            out.push(if lo0 <= v && v <= hi0 { 255 } else { 0 });
        }
    } else {
        for px in buf.chunks_exact(3) {
            let ok = lo[0] <= px[0]
                && px[0] <= hi[0]
                && lo[1] <= px[1]
                && px[1] <= hi[1]
                && lo[2] <= px[2]
                && px[2] <= hi[2];
            out.push(if ok { 255 } else { 0 });
        }
    }

    pyarray_from_vec(py, out, &[h, w])
}

#[pyfunction]
#[pyo3(signature = (src, mask, dst=None))]
fn copyTo(
    py: Python<'_>,
    src: PyReadonlyArrayDyn<u8>,
    mask: PyReadonlyArrayDyn<u8>,
    dst: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyArrayDyn<u8>>> {
    // Minimal `copyTo` with mask for uint8.
    // Mask is single-channel and broadcast across src channels.
    let (src_buf, src_shape) = py_image_ndarray_to_vec_and_shape(&src)?;
    let (mask_buf, mask_shape) = py_u8_ndarray_to_vec_and_shape(&mask)?;

    let (h, w, channels) = match src_shape.len() {
        2 => (src_shape[0], src_shape[1], 1usize),
        3 => (src_shape[0], src_shape[1], src_shape[2]),
        _ => return Err(PyValueError::new_err("unsupported src dims")),
    };
    if !(channels == 1 || channels == 3) {
        return Err(PyValueError::new_err("expected src with 1 or 3 channels"));
    }

    // Accept mask as (H,W) or (H,W,1)
    let (mh, mw) = match mask_shape.as_slice() {
        [mh, mw] => (*mh, *mw),
        [mh, mw, 1] => (*mh, *mw),
        _ => return Err(PyValueError::new_err("expected mask shape (H,W) or (H,W,1)")),
    };
    if mh != h || mw != w {
        return Err(PyValueError::new_err("mask must match src height/width"));
    }

    // Base output: dst if provided, else zeros.
    let mut out_buf: Vec<u8> = if let Some(dst_obj) = dst {
        if dst_obj.is_none() {
            vec![0u8; src_buf.len()]
        } else {
            let dst_arr = dst_obj
                .downcast::<PyArrayDyn<u8>>()
                .map_err(|_| PyTypeError::new_err("dst must be a uint8 numpy array"))?;
            let dst_shape = dst_arr.shape().to_vec();
            ensure_same_shape(&dst_shape, &src_shape)?;
            dst_arr.readonly().as_array().to_owned().into_raw_vec()
        }
    } else {
        vec![0u8; src_buf.len()]
    };

    if channels == 1 {
        for i in 0..(h * w) {
            if mask_buf[i] != 0 {
                out_buf[i] = src_buf[i];
            }
        }
    } else {
        for i in 0..(h * w) {
            if mask_buf[i] != 0 {
                let base = i * 3;
                out_buf[base] = src_buf[base];
                out_buf[base + 1] = src_buf[base + 1];
                out_buf[base + 2] = src_buf[base + 2];
            }
        }
    }

    // Write back to dst if provided; otherwise return new.
    if let Some(dst_obj) = dst {
        if dst_obj.is_none() {
            return pyarray_from_vec(py, out_buf, &src_shape);
        }
        let dst_arr = dst_obj
            .downcast::<PyArrayDyn<u8>>()
            .map_err(|_| PyTypeError::new_err("dst must be a uint8 numpy array"))?
            .to_owned();
        let mut rw = dst_arr
            .try_readwrite()
            .map_err(|_| PyValueError::new_err("dst is not writable"))?;
        write_vec_to_pyarray(&mut rw, &out_buf, &src_shape)?;
        return Ok(dst_arr.unbind());
    }

    pyarray_from_vec(py, out_buf, &src_shape)
}

#[pyfunction]
#[pyo3(signature = (src, ksize, dst=None))]
fn blur(
    py: Python<'_>,
    src: PyReadonlyArrayDyn<u8>,
    ksize: (i32, i32),
    dst: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyArrayDyn<u8>>> {
    let (buf, shape) = py_image_ndarray_to_vec_and_shape(&src)?;
    let (h, w, c) = parse_image_shape(&shape)?;
    let (kw, kh) = (ksize.0, ksize.1);
    if kw <= 0 || kh <= 0 {
        return Err(PyValueError::new_err("ksize must be positive"));
    }
    let kx = box_kernel_1d(kw as usize)?;
    let ky = box_kernel_1d(kh as usize)?;

    let dst_arr = parse_optional_dst_u8(dst)?;
    if let Some(ref d) = dst_arr {
        ensure_same_shape(d.shape(), &shape)?;
    }

    let out = convolve_separable_u8(&buf, h, w, c, &kx, &ky);
    write_or_return_u8(py, out, &shape, dst_arr)
}

#[pyfunction]
#[pyo3(signature = (src, ksize, sigmaX, dst=None, sigmaY=0.0, borderType=BORDER_DEFAULT))]
fn GaussianBlur(
    py: Python<'_>,
    src: PyReadonlyArrayDyn<u8>,
    ksize: (i32, i32),
    sigmaX: f64,
    dst: Option<&Bound<'_, PyAny>>,
    sigmaY: f64,
    borderType: i32,
) -> PyResult<Py<PyArrayDyn<u8>>> {
    // borderType is accepted for API compatibility but only BORDER_DEFAULT is supported.
    let _ = borderType;

    let (buf, shape) = py_image_ndarray_to_vec_and_shape(&src)?;
    let (h, w, c) = parse_image_shape(&shape)?;

    let (mut kw, mut kh) = (ksize.0, ksize.1);
    let sigma_y = if sigmaY == 0.0 { sigmaX } else { sigmaY };

    // If ksize is (0,0), approximate OpenCV's automatic size selection.
    if kw == 0 && kh == 0 {
        if sigmaX <= 0.0 {
            return Err(PyValueError::new_err("sigmaX must be > 0 when ksize is (0,0)"));
        }
        let k = ((sigmaX * 6.0).round() as i32) | 1;
        kw = k;
        kh = k;
    }
    if kw <= 0 || kh <= 0 || kw % 2 == 0 || kh % 2 == 0 {
        return Err(PyValueError::new_err("ksize must be positive odd"));
    }

    let kx = gaussian_kernel_1d(kw as usize, sigmaX)?;
    let ky = gaussian_kernel_1d(kh as usize, sigma_y)?;

    let dst_arr = parse_optional_dst_u8(dst)?;
    if let Some(ref d) = dst_arr {
        ensure_same_shape(d.shape(), &shape)?;
    }
    let out = convolve_separable_u8(&buf, h, w, c, &kx, &ky);
    write_or_return_u8(py, out, &shape, dst_arr)
}

fn pyarray_from_vec_i16(py: Python<'_>, data: Vec<i16>, shape: &[usize]) -> PyResult<PyObject> {
    let arr = ArrayD::from_shape_vec(IxDyn(shape), data)
        .map_err(|e| PyValueError::new_err(format!("failed to create ndarray: {e}")))?;
    Ok(arr.into_pyarray_bound(py).to_object(py))
}

fn pyarray_from_vec_f32(py: Python<'_>, data: Vec<f32>, shape: &[usize]) -> PyResult<PyObject> {
    let arr = ArrayD::from_shape_vec(IxDyn(shape), data)
        .map_err(|e| PyValueError::new_err(format!("failed to create ndarray: {e}")))?;
    Ok(arr.into_pyarray_bound(py).to_object(py))
}

fn pyarray_from_vec_f64(py: Python<'_>, data: Vec<f64>, shape: &[usize]) -> PyResult<PyObject> {
    let arr = ArrayD::from_shape_vec(IxDyn(shape), data)
        .map_err(|e| PyValueError::new_err(format!("failed to create ndarray: {e}")))?;
    Ok(arr.into_pyarray_bound(py).to_object(py))
}

fn write_to_dst_i16(dst_obj: &Bound<'_, PyAny>, data: &[i16], shape: &[usize]) -> PyResult<PyObject> {
    let dst = dst_obj
        .downcast::<PyArrayDyn<i16>>()
        .map_err(|_| PyTypeError::new_err("dst must be an int16 numpy array"))?
        .to_owned();
    ensure_same_shape(dst.shape(), shape)?;
    let mut rw = dst
        .try_readwrite()
        .map_err(|_| PyValueError::new_err("dst is not writable"))?;
    let src_view = ArrayViewD::from_shape(IxDyn(shape), data)
        .map_err(|e| PyValueError::new_err(format!("invalid shape: {e}")))?;
    rw.as_array_mut().assign(&src_view);
    Ok(dst.unbind().to_object(dst_obj.py()))
}

fn write_to_dst_f32(dst_obj: &Bound<'_, PyAny>, data: &[f32], shape: &[usize]) -> PyResult<PyObject> {
    let dst = dst_obj
        .downcast::<PyArrayDyn<f32>>()
        .map_err(|_| PyTypeError::new_err("dst must be a float32 numpy array"))?
        .to_owned();
    ensure_same_shape(dst.shape(), shape)?;
    let mut rw = dst
        .try_readwrite()
        .map_err(|_| PyValueError::new_err("dst is not writable"))?;
    let src_view = ArrayViewD::from_shape(IxDyn(shape), data)
        .map_err(|e| PyValueError::new_err(format!("invalid shape: {e}")))?;
    rw.as_array_mut().assign(&src_view);
    Ok(dst.unbind().to_object(dst_obj.py()))
}

fn write_to_dst_f64(dst_obj: &Bound<'_, PyAny>, data: &[f64], shape: &[usize]) -> PyResult<PyObject> {
    let dst = dst_obj
        .downcast::<PyArrayDyn<f64>>()
        .map_err(|_| PyTypeError::new_err("dst must be a float64 numpy array"))?
        .to_owned();
    ensure_same_shape(dst.shape(), shape)?;
    let mut rw = dst
        .try_readwrite()
        .map_err(|_| PyValueError::new_err("dst is not writable"))?;
    let src_view = ArrayViewD::from_shape(IxDyn(shape), data)
        .map_err(|e| PyValueError::new_err(format!("invalid shape: {e}")))?;
    rw.as_array_mut().assign(&src_view);
    Ok(dst.unbind().to_object(dst_obj.py()))
}

#[pyfunction]
#[pyo3(signature = (src, ddepth, dx, dy, dst=None, ksize=3, scale=1.0, delta=0.0, borderType=BORDER_DEFAULT))]
fn Sobel(
    py: Python<'_>,
    src: PyReadonlyArrayDyn<u8>,
    ddepth: i32,
    dx: i32,
    dy: i32,
    dst: Option<&Bound<'_, PyAny>>,
    ksize: i32,
    scale: f64,
    delta: f64,
    borderType: i32,
) -> PyResult<PyObject> {
    if borderType != BORDER_DEFAULT {
        return Err(PyValueError::new_err("only BORDER_DEFAULT is supported"));
    }
    if ksize != 3 {
        return Err(PyValueError::new_err("only ksize=3 is supported"));
    }
    if !(dx == 0 || dx == 1) || !(dy == 0 || dy == 1) || (dx == 0 && dy == 0) {
        return Err(PyValueError::new_err("only dx/dy in {0,1} (not both 0) are supported"));
    }
    if !scale.is_finite() || !delta.is_finite() {
        return Err(PyValueError::new_err("scale/delta must be finite"));
    }

    let (buf, shape) = py_image_ndarray_to_vec_and_shape(&src)?;
    let (h, w, c) = parse_image_shape(&shape)?;

    let deriv = [-1i32, 0i32, 1i32];
    let smooth = [1i32, 2i32, 1i32];
    let kx: &[i32] = if dx == 1 { &deriv } else { &smooth };
    let ky: &[i32] = if dy == 1 { &deriv } else { &smooth };

    let raw = convolve_separable_i32(&buf, h, w, c, kx, ky);
    let scale = scale;
    let delta = delta;

    match ddepth {
        -1 | CV_8U => {
            let out: Vec<u8> = raw
                .iter()
                .map(|&v| (v as f64 * scale + delta).round().clamp(0.0, 255.0) as u8)
                .collect();
            let dst_arr = parse_optional_dst_u8(dst)?;
            if let Some(ref d) = dst_arr {
                ensure_same_shape(d.shape(), &shape)?;
            }
            Ok(write_or_return_u8(py, out, &shape, dst_arr)?.to_object(py))
        }
        CV_16S => {
            let out: Vec<i16> = raw
                .iter()
                .map(|&v| {
                    let vv = (v as f64 * scale + delta).round();
                    vv.clamp(i16::MIN as f64, i16::MAX as f64) as i16
                })
                .collect();
            if let Some(dst_obj) = dst {
                if dst_obj.is_none() {
                    return pyarray_from_vec_i16(py, out, &shape);
                }
                return write_to_dst_i16(dst_obj, &out, &shape);
            }
            pyarray_from_vec_i16(py, out, &shape)
        }
        CV_32F => {
            let out: Vec<f32> = raw
                .iter()
                .map(|&v| (v as f64 * scale + delta) as f32)
                .collect();
            if let Some(dst_obj) = dst {
                if dst_obj.is_none() {
                    return pyarray_from_vec_f32(py, out, &shape);
                }
                return write_to_dst_f32(dst_obj, &out, &shape);
            }
            pyarray_from_vec_f32(py, out, &shape)
        }
        CV_64F => {
            let out: Vec<f64> = raw.iter().map(|&v| v as f64 * scale + delta).collect();
            if let Some(dst_obj) = dst {
                if dst_obj.is_none() {
                    return pyarray_from_vec_f64(py, out, &shape);
                }
                return write_to_dst_f64(dst_obj, &out, &shape);
            }
            pyarray_from_vec_f64(py, out, &shape)
        }
        _ => Err(PyValueError::new_err("unsupported ddepth")),
    }
}

#[pyfunction]
#[pyo3(signature = (src, ddepth, dx, dy, dst=None, scale=1.0, delta=0.0, borderType=BORDER_DEFAULT))]
fn Scharr(
    py: Python<'_>,
    src: PyReadonlyArrayDyn<u8>,
    ddepth: i32,
    dx: i32,
    dy: i32,
    dst: Option<&Bound<'_, PyAny>>,
    scale: f64,
    delta: f64,
    borderType: i32,
) -> PyResult<PyObject> {
    if borderType != BORDER_DEFAULT {
        return Err(PyValueError::new_err("only BORDER_DEFAULT is supported"));
    }
    if !((dx == 1 && dy == 0) || (dx == 0 && dy == 1)) {
        return Err(PyValueError::new_err("Scharr supports (dx,dy) of (1,0) or (0,1)"));
    }
    if !scale.is_finite() || !delta.is_finite() {
        return Err(PyValueError::new_err("scale/delta must be finite"));
    }

    let (buf, shape) = py_image_ndarray_to_vec_and_shape(&src)?;
    let (h, w, c) = parse_image_shape(&shape)?;

    let deriv = [-1i32, 0i32, 1i32];
    let smooth = [3i32, 10i32, 3i32];
    let kx: &[i32] = if dx == 1 { &deriv } else { &smooth };
    let ky: &[i32] = if dy == 1 { &deriv } else { &smooth };

    let raw = convolve_separable_i32(&buf, h, w, c, kx, ky);

    match ddepth {
        -1 | CV_8U => {
            let out: Vec<u8> = raw
                .iter()
                .map(|&v| (v as f64 * scale + delta).round().clamp(0.0, 255.0) as u8)
                .collect();
            let dst_arr = parse_optional_dst_u8(dst)?;
            if let Some(ref d) = dst_arr {
                ensure_same_shape(d.shape(), &shape)?;
            }
            Ok(write_or_return_u8(py, out, &shape, dst_arr)?.to_object(py))
        }
        CV_16S => {
            let out: Vec<i16> = raw
                .iter()
                .map(|&v| {
                    let vv = (v as f64 * scale + delta).round();
                    vv.clamp(i16::MIN as f64, i16::MAX as f64) as i16
                })
                .collect();
            if let Some(dst_obj) = dst {
                if dst_obj.is_none() {
                    return pyarray_from_vec_i16(py, out, &shape);
                }
                return write_to_dst_i16(dst_obj, &out, &shape);
            }
            pyarray_from_vec_i16(py, out, &shape)
        }
        CV_32F => {
            let out: Vec<f32> = raw
                .iter()
                .map(|&v| (v as f64 * scale + delta) as f32)
                .collect();
            if let Some(dst_obj) = dst {
                if dst_obj.is_none() {
                    return pyarray_from_vec_f32(py, out, &shape);
                }
                return write_to_dst_f32(dst_obj, &out, &shape);
            }
            pyarray_from_vec_f32(py, out, &shape)
        }
        CV_64F => {
            let out: Vec<f64> = raw.iter().map(|&v| v as f64 * scale + delta).collect();
            if let Some(dst_obj) = dst {
                if dst_obj.is_none() {
                    return pyarray_from_vec_f64(py, out, &shape);
                }
                return write_to_dst_f64(dst_obj, &out, &shape);
            }
            pyarray_from_vec_f64(py, out, &shape)
        }
        _ => Err(PyValueError::new_err("unsupported ddepth")),
    }
}

#[pyfunction]
#[pyo3(signature = (image, threshold1, threshold2, edges=None, apertureSize=3, L2gradient=false))]
fn Canny(
    py: Python<'_>,
    image: PyReadonlyArrayDyn<u8>,
    threshold1: f64,
    threshold2: f64,
    edges: Option<&Bound<'_, PyAny>>,
    apertureSize: i32,
    L2gradient: bool,
) -> PyResult<Py<PyArrayDyn<u8>>> {
    if !threshold1.is_finite() || !threshold2.is_finite() {
        return Err(PyValueError::new_err("thresholds must be finite"));
    }
    if apertureSize != 3 && apertureSize != 5 {
        return Err(PyValueError::new_err("only apertureSize 3 or 5 is supported"));
    }

    let (buf, shape) = py_image_ndarray_to_vec_and_shape(&image)?;
    let (h, w, channels) = match shape.as_slice() {
        [h, w] => (*h, *w, 1usize),
        [h, w, 1] => (*h, *w, 1usize),
        [h, w, 3] => (*h, *w, 3usize),
        _ => {
            return Err(PyValueError::new_err(
                "Canny expects (H,W), (H,W,1), or (H,W,3) uint8 image",
            ))
        }
    };

    // Canny always returns a single-channel edge map.
    let out_shape = vec![h, w];
    if h < 3 || w < 3 {
        // OpenCV returns an empty edge map for tiny images.
        let dst_arr = parse_optional_dst_u8(edges)?;
        if let Some(ref d) = dst_arr {
            ensure_same_shape(d.shape(), &out_shape)?;
        }
        return write_or_return_u8(py, vec![0u8; h * w], &out_shape, dst_arr);
    }

    let (mut low, mut high) = (threshold1, threshold2);
    if low > high {
        std::mem::swap(&mut low, &mut high);
    }

    // Gradients
    // Convert to grayscale if needed.
    let gray: Vec<u8> = if channels == 3 {
        bgr_to_gray(&buf)
    } else {
        buf
    };

    let (kx_deriv, ky_smooth): (Vec<i32>, Vec<i32>) = if apertureSize == 3 {
        (vec![-1, 0, 1], vec![1, 2, 1])
    } else {
        (vec![-1, -2, 0, 2, 1], vec![1, 4, 6, 4, 1])
    };
    let gx = convolve_separable_i32(&gray, h, w, 1, &kx_deriv, &ky_smooth);
    let gy = convolve_separable_i32(&gray, h, w, 1, &ky_smooth, &kx_deriv);

    // Magnitude
    let mut mag = vec![0f32; h * w];
    for i in 0..(h * w) {
        let x = gx[i] as f32;
        let y = gy[i] as f32;
        mag[i] = if L2gradient {
            (x * x + y * y).sqrt()
        } else {
            x.abs() + y.abs()
        };
    }

    // Non-maximum suppression
    let mut nms = vec![0f32; h * w];
    let t1 = 0.41421356f32; // tan(22.5)
    let t2 = 2.41421356f32; // tan(67.5)

    for y in 1..(h - 1) {
        for x in 1..(w - 1) {
            let idx = y * w + x;
            let gxv = gx[idx] as f32;
            let gyv = gy[idx] as f32;
            let a = gxv.abs();
            let b = gyv.abs();

            let (n1, n2) = if b <= a * t1 {
                // 0 deg
                (idx - 1, idx + 1)
            } else if b >= a * t2 {
                // 90 deg
                (idx - w, idx + w)
            } else if gxv * gyv > 0.0 {
                // 45 deg
                (idx - w - 1, idx + w + 1)
            } else {
                // 135 deg
                (idx - w + 1, idx + w - 1)
            };

            let m = mag[idx];
            if m >= mag[n1] && m >= mag[n2] {
                nms[idx] = m;
            }
        }
    }

    // Double threshold + hysteresis
    let high_t = high as f32;
    let low_t = low as f32;
    let mut out = vec![0u8; h * w];
    let mut stack = Vec::<usize>::new();

    for y in 1..(h - 1) {
        for x in 1..(w - 1) {
            let idx = y * w + x;
            if nms[idx] >= high_t {
                out[idx] = 255;
                stack.push(idx);
            }
        }
    }

    while let Some(idx) = stack.pop() {
        let y = idx / w;
        let x = idx % w;
        for yy in (y.saturating_sub(1))..=((y + 1).min(h - 1)) {
            for xx in (x.saturating_sub(1))..=((x + 1).min(w - 1)) {
                let nidx = yy * w + xx;
                if out[nidx] == 0 && nms[nidx] >= low_t {
                    out[nidx] = 255;
                    stack.push(nidx);
                }
            }
        }
    }

    // Ensure borders are zeroed.
    for x in 0..w {
        out[x] = 0;
        out[(h - 1) * w + x] = 0;
    }
    for y in 0..h {
        out[y * w] = 0;
        out[y * w + (w - 1)] = 0;
    }

    let dst_arr = parse_optional_dst_u8(edges)?;
    if let Some(ref d) = dst_arr {
        ensure_same_shape(d.shape(), &out_shape)?;
    }
    write_or_return_u8(py, out, &out_shape, dst_arr)
}

#[pyfunction]
#[pyo3(signature = (src, code))]
fn cvtColor(py: Python<'_>, src: PyReadonlyArrayDyn<u8>, code: i32) -> PyResult<Py<PyArrayDyn<u8>>> {
    let (buf, shape) = py_image_ndarray_to_vec_and_shape(&src)?;
    match code {
        COLOR_BGR2RGB => {
            if shape.len() != 3 || shape[2] != 3 {
                return Err(PyValueError::new_err("expected (H,W,3) array"));
            }
            let mut out = buf;
            rgb_to_bgr_inplace(&mut out);
            pyarray_from_vec(py, out, &[shape[0], shape[1], 3])
        }
        // In OpenCV, COLOR_RGB2BGR == COLOR_BGR2RGB (same numeric constant).
        COLOR_BGR2GRAY => {
            if shape.len() != 3 || shape[2] != 3 {
                return Err(PyValueError::new_err("expected (H,W,3) BGR array"));
            }
            let gray = bgr_to_gray(&buf);
            pyarray_from_vec(py, gray, &[shape[0], shape[1]])
        }
        COLOR_GRAY2BGR => {
            if shape.len() == 2 || (shape.len() == 3 && shape[2] == 1) {
                let gray = if shape.len() == 3 { buf } else { buf };
                let bgr = gray_to_bgr(&gray);
                pyarray_from_vec(py, bgr, &[shape[0], shape[1], 3])
            } else {
                Err(PyValueError::new_err("expected (H,W) or (H,W,1) grayscale array"))
            }
        }
        _ => Err(PyValueError::new_err("unsupported color conversion code")),
    }
}

#[pyfunction]
#[pyo3(signature = (src, dsize, interpolation=INTER_LINEAR))]
fn resize(
    py: Python<'_>,
    src: PyReadonlyArrayDyn<u8>,
    dsize: (i32, i32),
    interpolation: i32,
) -> PyResult<Py<PyArrayDyn<u8>>> {
    if dsize.0 <= 0 || dsize.1 <= 0 {
        return Err(PyValueError::new_err("dsize must be positive (width, height)"));
    }
    let (buf, shape) = py_image_ndarray_to_vec_and_shape(&src)?;
    let filter = filter_from_interpolation(interpolation);
    match shape.len() {
        2 => {
            let src_img = image::GrayImage::from_raw(shape[1] as u32, shape[0] as u32, buf)
                .ok_or_else(|| PyValueError::new_err("invalid grayscale shape"))?;
            let dst = DynamicImage::ImageLuma8(src_img)
                .resize_exact(dsize.0 as u32, dsize.1 as u32, filter)
                .to_luma8();
            pyarray_from_vec(py, dst.into_raw(), &[dsize.1 as usize, dsize.0 as usize])
        }
        3 => {
            let c = shape[2];
            if c == 1 {
                let src_img = image::GrayImage::from_raw(shape[1] as u32, shape[0] as u32, buf)
                    .ok_or_else(|| PyValueError::new_err("invalid grayscale shape"))?;
                let dst = DynamicImage::ImageLuma8(src_img)
                    .resize_exact(dsize.0 as u32, dsize.1 as u32, filter)
                    .to_luma8();
                pyarray_from_vec(py, dst.into_raw(), &[dsize.1 as usize, dsize.0 as usize, 1])
            } else if c == 3 {
                // BGR -> RGB for image crate, resize, then back to BGR
                let mut rgb = buf;
                rgb_to_bgr_inplace(&mut rgb);
                let src_img = image::RgbImage::from_raw(shape[1] as u32, shape[0] as u32, rgb)
                    .ok_or_else(|| PyValueError::new_err("invalid color shape"))?;
                let mut dst = DynamicImage::ImageRgb8(src_img)
                    .resize_exact(dsize.0 as u32, dsize.1 as u32, filter)
                    .to_rgb8()
                    .into_raw();
                // RGB -> BGR for return
                rgb_to_bgr_inplace(&mut dst);
                pyarray_from_vec(py, dst, &[dsize.1 as usize, dsize.0 as usize, 3])
            } else {
                Err(PyValueError::new_err("expected 1 or 3 channels"))
            }
        }
        _ => Err(PyValueError::new_err("unexpected dims")),
    }
}

#[pymodule]
fn cv2(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // functions
    m.add_function(wrap_pyfunction!(imread, m)?)?;
    m.add_function(wrap_pyfunction!(imwrite, m)?)?;
    m.add_function(wrap_pyfunction!(imdecode, m)?)?;
    m.add_function(wrap_pyfunction!(imencode, m)?)?;
    m.add_function(wrap_pyfunction!(cvtColor, m)?)?;
    m.add_function(wrap_pyfunction!(resize, m)?)?;
    m.add_function(wrap_pyfunction!(norm, m)?)?;
    m.add_function(wrap_pyfunction!(absdiff, m)?)?;
    m.add_function(wrap_pyfunction!(add, m)?)?;
    m.add_function(wrap_pyfunction!(subtract, m)?)?;
    m.add_function(wrap_pyfunction!(split, m)?)?;
    m.add_function(wrap_pyfunction!(merge, m)?)?;
    m.add_function(wrap_pyfunction!(bitwise_not, m)?)?;
    m.add_function(wrap_pyfunction!(bitwise_and, m)?)?;
    m.add_function(wrap_pyfunction!(bitwise_or, m)?)?;
    m.add_function(wrap_pyfunction!(bitwise_xor, m)?)?;
    m.add_function(wrap_pyfunction!(threshold, m)?)?;
    m.add_function(wrap_pyfunction!(countNonZero, m)?)?;
    m.add_function(wrap_pyfunction!(compare, m)?)?;
    m.add_function(wrap_pyfunction!(inRange, m)?)?;
    m.add_function(wrap_pyfunction!(copyTo, m)?)?;
    m.add_function(wrap_pyfunction!(blur, m)?)?;
    m.add_function(wrap_pyfunction!(GaussianBlur, m)?)?;
    m.add_function(wrap_pyfunction!(Sobel, m)?)?;
    m.add_function(wrap_pyfunction!(Scharr, m)?)?;
    m.add_function(wrap_pyfunction!(Canny, m)?)?;
    m.add_function(wrap_pyfunction!(flip, m)?)?;
    m.add_function(wrap_pyfunction!(transpose, m)?)?;
    m.add_function(wrap_pyfunction!(rotate, m)?)?;

    // constants
    m.add("IMREAD_GRAYSCALE", IMREAD_GRAYSCALE)?;
    m.add("IMREAD_COLOR", IMREAD_COLOR)?;
    m.add("IMREAD_UNCHANGED", IMREAD_UNCHANGED)?;

    m.add("NORM_INF", NORM_INF)?;
    m.add("NORM_L1", NORM_L1)?;
    m.add("NORM_L2", NORM_L2)?;

    m.add("THRESH_BINARY", THRESH_BINARY)?;
    m.add("THRESH_BINARY_INV", THRESH_BINARY_INV)?;
    m.add("THRESH_TRUNC", THRESH_TRUNC)?;
    m.add("THRESH_TOZERO", THRESH_TOZERO)?;
    m.add("THRESH_TOZERO_INV", THRESH_TOZERO_INV)?;

    m.add("CMP_EQ", CMP_EQ)?;
    m.add("CMP_GT", CMP_GT)?;
    m.add("CMP_GE", CMP_GE)?;
    m.add("CMP_LT", CMP_LT)?;
    m.add("CMP_LE", CMP_LE)?;
    m.add("CMP_NE", CMP_NE)?;

    m.add("BORDER_DEFAULT", BORDER_DEFAULT)?;

    m.add("CV_8U", CV_8U)?;
    m.add("CV_16S", CV_16S)?;
    m.add("CV_32F", CV_32F)?;
    m.add("CV_64F", CV_64F)?;

    m.add("ROTATE_90_CLOCKWISE", ROTATE_90_CLOCKWISE)?;
    m.add("ROTATE_180", ROTATE_180)?;
    m.add("ROTATE_90_COUNTERCLOCKWISE", ROTATE_90_COUNTERCLOCKWISE)?;

    m.add("COLOR_BGR2RGB", COLOR_BGR2RGB)?;
    m.add("COLOR_RGB2BGR", COLOR_RGB2BGR)?;
    m.add("COLOR_BGR2GRAY", COLOR_BGR2GRAY)?;
    m.add("COLOR_GRAY2BGR", COLOR_GRAY2BGR)?;

    m.add("INTER_NEAREST", INTER_NEAREST)?;
    m.add("INTER_LINEAR", INTER_LINEAR)?;
    m.add("INTER_CUBIC", INTER_CUBIC)?;
    m.add("INTER_AREA", INTER_AREA)?;
    m.add("INTER_LANCZOS4", INTER_LANCZOS4)?;

    // module docstring
    m.add("__doc__", "Rust-native minimal subset of cv2 (crabvision)")?;

    // Provide a cheap version string for debugging
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    Ok(())
}
