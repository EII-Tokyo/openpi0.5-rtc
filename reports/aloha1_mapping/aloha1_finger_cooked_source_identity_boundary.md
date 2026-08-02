# ALOHA1 finger cooked-source identity boundary

- Status: **PASS_SOURCE_MISMATCH_DETECTED**
- Classification: **LEGACY_COOKED_SOURCE_NOT_CURRENT_SUPPLIER_SOURCE**
- Next gate: **REQUIRES_SUPPLIER_CAD_COOKED_READBACK**
- Isaac runtime started: `false`
- Final/default asset modified: `false`

| Side | Supplier CAD SHA-256 | Legacy cooked source SHA-256 | Supplier faces | Legacy faces | Legacy/supplier volume |
|---|---|---|---:|---:|---:|
| left | `c6710d0fe5b2030a32722d9df5c0b553c771c9d61d92b8ddaec36c94c5963488` | `df73ae5b9058e5d50a6409ac2ab687dade75053a86591bb5e23ab051dbf2d659` | 1662 | 1666 | 0.78313345 |
| right | `b0979c5d55fee448dab512dc75b1251bab17d94892decd01de9a6e76c01482d1` | `56fb3cc1236d4193106038adf8e457c7252ae9e86c7cee6dabf0578c53666358` | 1662 | 1666 | 0.783147068 |

The saved 32-piece cooked decomposition belongs to the recorded historical gym-aloha STL inputs, not to the current supplier-assembly B-Rep tessellations. It cannot certify the supplier-CAD inward contact surfaces. A new isolated Isaac Sim 5.1 cooked-geometry readback is required; no legacy runtime result is promoted across this source boundary.
