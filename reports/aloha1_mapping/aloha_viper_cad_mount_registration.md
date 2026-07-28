# ALOHA Viper supplier-CAD mounting datum registration

- Status: `PASS`
- Method: `CONTROLLED_ORTHOGONAL_PLANAR_DATUM_REGISTRATION`
- Threshold: `0.0002000 m`
- Full-surface ICP used for decision: `false`
- Physical measurement: `false`

| Datum | CAD coordinate (m) | Stage coordinate (m) | Residual (m) | CAD triangles/area | Stage triangles/area | Status |
|---|---:|---:|---:|---:|---:|---:|
| `x_min` | -1.19409512833e-08 | 0 | 1.19409512833e-08 | 34 / 0.000297133716819 m² | 79 / 0.00167885523515 m² | `PASS` |
| `y_min` | -0.0750000025756 | -0.0749990005493 | 1.0020262359e-06 | 6 / 0.000847327972412 m² | 30 / 0.000894514166958 m² | `PASS` |
| `y_max` | 0.0749999974244 | 0.0750009994507 | 1.00202623592e-06 | 6 / 0.000847327972412 m² | 38 / 0.000894565351012 m² | `PASS` |
| `z_max` | 0.0352500024736 | 0.03525 | 2.47363801664e-09 | 2 / 0.000101661369324 m² | 10 / 0.0011045 m² | `PASS` |

This report validates a controlled supplier-CAD-to-Stage datum registration. It is not a physical measurement and does not validate collision, contact, or grasping.
