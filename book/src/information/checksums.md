# Checksums and hashes: CRC-32 and SHA-256

<div class="entry-summary">

| | |
|---|---|
| **Problem** | Detect accidental corruption of data cheaply (a checksum), or fingerprint data so that no one can find two inputs with the same fingerprint (a cryptographic hash) |
| **Devised by** | W. Wesley Peterson (1961), cyclic redundancy checks; the IEEE 802.3 committee (1975–83), the CRC-32 polynomial; Guy Castagnoli and colleagues (1993), CRC-32C; the US National Security Agency (2001), SHA-256 |
| **Accuracy** | Exact: equal to Python's `zlib.crc32` and `hashlib.sha256` on every test input, including the FIPS long vector (table below) |
| **Cost** | CRC: one table lookup per byte (about 330 MB/s here); SHA-256: 64 rounds per 64-byte block (about 80–100 MB/s) |
| **API** | [`crc32`](../../api/lib_jsl_information/operators/crc32/fn.crc32.html), [`crc32c`](../../api/lib_jsl_information/operators/crc32/fn.crc32c.html), [`Crc32`](../../api/lib_jsl_information/operators/crc32/struct.Crc32.html) (`new`, `castagnoli`, `update`, `finalize`), [`sha256`](../../api/lib_jsl_information/operators/sha256/fn.sha256.html), [`Sha256`](../../api/lib_jsl_information/operators/sha256/struct.Sha256.html) (`new`, `update`, `finalize`), [`to_hex`](../../api/lib_jsl_information/operators/sha256/fn.to_hex.html) |

</div>

## Overview

A **cyclic redundancy check** (CRC) treats a message as a polynomial over GF(2) and appends the remainder of its division by a fixed generator polynomial. Bit errors that do not leave the message divisible are detected, and the generator is chosen so that the common ones (single bits, pairs, short bursts) never do. CRCs are linear and cheap,
but easy to forge. A **cryptographic hash** such as **SHA-256** maps any input to 256 bits so that no feasible computation can find an input with a given digest, or two inputs with the same one. It is used for signatures, content addressing and integrity against deliberate tampering.

## History

W. Wesley Peterson (1961) proposed cyclic codes for error detection and showed how a shift register computes the remainder. The 32-bit polynomial 0x04C11DB7 was chosen for Ethernet in the 1970s, following work by Joseph Hammond and colleagues, and went on to zip, gzip and PNG. Guy Castagnoli, Stefan Bräuer and Martin
Herrmann (1993) searched for 32-bit polynomials with better distance properties; their CRC-32C is used in iSCSI, SCTP and ext4, and has its own x86 instruction. SHA-256 was designed by the NSA and published by NIST in 2001 (FIPS 180-2) after weaknesses were found in MD5 and SHA-1. It is a Merkle-Damgård construction around a block cipher-like compression function and remains unbroken.

## The idea

### CRC: division over GF(2)

A **checksum** is a short value computed from a message and sent along with it. The receiver
recomputes it, and a mismatch reveals that the message was corrupted. A **cyclic redundancy
check** (CRC) is a checksum defined by polynomial division, which gives it guaranteed detection
properties that a simple sum of the bytes lacks.

**The definition.** Read the message's bits as the coefficients of a polynomial $M(x)$, with
arithmetic on the coefficients modulo 2 (so addition and subtraction are both XOR). Fix a
**generator polynomial** $G(x)$ of degree 32. The CRC is the remainder

$$
R(x) = x^{32}M(x) \bmod G(x),
$$

a polynomial of degree below 32, that is, 32 bits. Multiplying by $x^{32}$ shifts the message up
to make room, and the transmitted word $x^{32}M(x) + R(x)$ is the message followed by its CRC.
Since the remainder has been added back, that word is an exact multiple of $G$. (This is the
same construction as the [systematic Reed-Solomon code](../coding/reed-solomon.md#systematic-encoding),
over the field of two elements.)

**What it detects.** Corruption adds an error polynomial $E(x)$, with a 1 at each flipped bit.
The receiver finds a multiple of $G$ plus $E$, and the check fails unless $G$ happens to divide
$E$. So an error goes undetected only if its pattern is a multiple of the generator. That gives
guarantees:

- A single-bit error is $E = x^i$, which no $G$ with more than one term divides: all single-bit
  errors are caught.
- A burst of errors confined to 32 consecutive bits or fewer is $x^i$ times a polynomial of
  degree below 32, which $G$ (of degree 32, with a nonzero constant term) cannot divide: all
  such bursts are caught.
- Random corruption produces a random remainder, which is zero with probability $2^{-32}$. For CRC-32, all two- and three-bit errors are caught in messages up to about 11 kB.

{{#include ../generated/information/crc_detection.md}}

In practice the CRC is computed a byte at a time with a 256-entry table of remainders, in **reflected** bit order (least significant bit first, matching serial transmission). The register starts at all ones, so leading zero bytes are not invisible, and the result is inverted. The table for 0xEDB88320, the bit reversal of 0x04C11DB7, gives the standard CRC-32. {{#include ../generated/information/crc32c_facts.md}}

### SHA-256

A CRC protects against accidents, not against people: because it is linear, it is easy to alter a
message deliberately so that the CRC still matches. A **cryptographic hash function** is designed
so that this is infeasible. It maps a message of any length to a fixed-size **digest** in such a
way that nobody can find a message with a given digest, or two messages with the same digest,
faster than by trying an astronomical number of candidates. SHA-256 produces a 256-bit digest.

The message is padded with a 1 bit, zeros and its 64-bit length to a multiple of 512 bits. The 256-bit state, initially fixed constants, is updated by each 512-bit block. The block is expanded into 64 words with rotations and shifts. Then 64 rounds mix them into eight working variables, using the nonlinear choice and majority functions
$\mathrm{Ch}(e, f, g) = (e \wedge f) \oplus (\neg e \wedge g)$ and $\mathrm{Maj}(a, b, c)$, rotations, and 64 round constants from the cube roots of the first 64 primes. The result is added to the state. The final state is the digest.

A good hash shows the **avalanche** effect: changing one input bit changes each output bit with probability 1/2, independently:

![SHA-256 avalanche](../generated/information/avalanche.svg)

{{#include ../generated/information/checksum_reference.md}}

{{#include ../generated/information/throughput_facts.md}}

## Algorithm

**CRC-32**, a byte at a time.

*Table*, built once: for each of the 256 byte values, start with the value and repeat eight
times "shift right by one bit, and if the bit shifted out was 1, XOR with the reflected
polynomial 0xEDB88320". The result is the remainder contributed by that byte.

*Checksum*:

1. set the 32-bit state to all ones (0xFFFFFFFF);
2. for each byte of the message: XOR the byte into the low eight bits of the state; use those
   eight bits as an index into the table; the new state is the table entry XORed with the old
   state shifted right by eight bits;
3. return the state with all its bits inverted.

The cost is one table lookup per byte.

**SHA-256.** The state is eight 32-bit words, set to fixed constants at the start.

1. **Buffer** the input into blocks of 64 bytes, and compress each complete block into the
   state.
2. **Compress one block**: read it as sixteen 32-bit words and expand them to 64 by a
   recurrence of rotations, shifts and XORs. Copy the state into eight working variables.
   Perform 64 rounds, each of which combines the working variables with one expanded word and
   one round constant through the choice and majority functions, rotations and additions
   modulo $2^{32}$. Add the working variables into the state.
3. **Finish**: append a single 1 bit, then zeros, then the length of the message in bits as a
   64-bit number, so that the total is a multiple of 64 bytes. This makes one or two final
   blocks, which are compressed.
4. Output the eight state words, most significant byte first: the 32-byte digest.

## Implementation

```rust,ignore
{{#include ../../../crates/information/src/operators/crc32.rs:crc_update}}
```

```rust,ignore
{{#include ../../../crates/information/src/operators/sha256.rs:sha256_rounds}}
```

- Both `Crc32` and `Sha256` can be fed in pieces, and `finalize` leaves them usable, so a running checksum can be read at any point.
- The CRC tables are built once per polynomial, on first use.

## In practice

```rust,ignore
{{#include ../../../crates/information/examples/information_checksums.rs}}
```

```text
CRC-32  414fa339
CRC-32C 22620404
SHA-256 d7a8fbb307d7809469ca9abcb0082e4f8d5651e46d3cdb762d02d0bf37c9e592
streamed: 414fa339 d7a8fbb307d78094
"dog" -> "cog": CRC-32 4400b5bc, 123 of 256 digest bits changed
```

**Strengths**

- CRCs: fast, with guaranteed detection of the common error patterns; SHA-256: collision resistance with no known practical attack.
- Streaming interfaces; results identical to the standard implementations.

**Weaknesses**

- A CRC protects only against accidents: it is linear, so anyone can alter data and fix the CRC.
- These are portable software implementations, slower than the hardware CRC and SHA instructions of modern processors.

## Exercises

1. Show that a CRC with a generator divisible by $x + 1$ detects every error with an odd number of flipped bits.
2. Show that every burst of length at most $\deg G$ is detected if $G$ has a nonzero constant term.
3. Show that $\mathrm{CRC}(a \oplus b) = \mathrm{CRC}(a) \oplus \mathrm{CRC}(b) \oplus \mathrm{CRC}(0)$ for equal-length messages, and explain why this makes CRCs forgeable.
4. Estimate, by the birthday bound, how many random messages must be hashed before two share a 32-bit CRC, and two a SHA-256 digest.
5. Verify the padding rule: how many blocks does SHA-256 compress for messages of 55, 56 and 64 bytes?

## Further reading

- W. W. Peterson and D. T. Brown, "Cyclic codes for error detection", *Proceedings of the IRE* 49 (1961) 228–235.
- G. Castagnoli, S. Bräuer and M. Herrmann, "Optimization of cyclic redundancy-check codes with 24 and 32 parity bits", *IEEE Transactions on Communications* 41 (1993) 883–892.
- National Institute of Standards and Technology, *Secure Hash Standard (SHS)*, FIPS PUB 180-4, 2015.
- P. Koopman, "32-bit cyclic redundancy codes for Internet applications", *Proceedings of the International Conference on Dependable Systems and Networks* (2002) 459–468.
