| input | CRC-32 | vs `zlib.crc32` | SHA-256 (first 16 hex digits) | vs `hashlib.sha256` |
|---|---:|---:|---:|---:|
| "" | `00000000` | equal | `e3b0c44298fc1c14…` | equal |
| "abc" | `352441c2` | equal | `ba7816bf8f01cfea…` | equal |
| "123456789" | `cbf43926` | equal | `15e2b0d3c33891eb…` | equal |
| "The quick brown fox jumps over the lazy dog" | `414fa339` | equal | `d7a8fbb307d78094…` | equal |
| the Bible test text (4.6 MB) | `4f1cde02` | equal | `fe59fc0c2e6452cf…` | equal |
| a million "a"s |  |  | `cdc76e5c9914fb92…` | equal |
