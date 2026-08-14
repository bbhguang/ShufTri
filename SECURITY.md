# PSI implementation and measurement boundary

The backend implements a two-message commutative-DH/OPRF-style PSI-CA over
Ristretto255, a prime-order group in the Curve25519 family. Identifiers use a
domain-separated BLAKE2b-512 hash-to-group. Each party samples a fresh nonzero
scalar for every session with libsodium's OS-backed CSPRNG; response arrays are
privately shuffled, received frames and Ristretto points are validated, and
secret workspaces are cleared.

The security model is semi-honest. The initiator learns the intersection
cardinality; set sizes, message lengths, peer/session metadata, and timing are
visible. The implementation does not claim malicious security, DLEQ
verifiability, authentication, replay protection, or RFC 9497 VOPRF
conformance.

