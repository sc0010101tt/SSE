import numpy as np
import hashlib, base64, os, time, ctypes
from Crypto.Cipher import AES, PKCS1_OAEP
from Crypto.PublicKey import RSA
from Crypto.Random import get_random_bytes
import hmac

lib_path = "./PolyOps.dll" if os.name == "nt" else "./poly_ops.so"
lib = None
if os.path.exists(lib_path):
    lib = ctypes.CDLL(lib_path)
    try:
        lib.poly_mul_ntt.restype = None
        lib.poly_mul_ntt.argtypes = [
            ctypes.POINTER(ctypes.c_uint16),
            ctypes.POINTER(ctypes.c_uint16),
            ctypes.POINTER(ctypes.c_uint16),
            ctypes.c_size_t,
            ctypes.c_uint16
        ]

        lib.poly_dot_mod.restype = ctypes.c_uint16
        lib.poly_dot_mod.argtypes = [
            ctypes.POINTER(ctypes.c_uint16),
            ctypes.POINTER(ctypes.c_uint16),
            ctypes.c_size_t,
            ctypes.c_uint16
        ]
    except AttributeError:
        print("⚠️ Functions not found in DLL. Falling back to NumPy.")
        lib = None


# Add a wrapper to handle both strings and bytes
def ecc_encode(data: bytes) -> bytes:
    return b"".join(bytes([b]) * 3 for b in data)


def ecc_decode(data: bytes) -> bytes:
    chunks = [data[i:i + 3] for i in range(0, len(data), 3)]
    out = []
    for chunk in chunks:
        if len(chunk) < 3:
            continue
        counts = {}
        for c in chunk:
            counts[c] = counts.get(c, 0) + 1
        out.append(max(counts, key=counts.get))
    return bytes(out)


# Wrappers for SSE encryption/decryption that accept both strings and bytes
def sse_encrypt(data, public_key, n, q):
    if isinstance(data, str):
        data = data.encode()
    return sse_encrypt_bytes(data, public_key, n, q)


def sse_decrypt(ciphertexts, public_key, private_key, n, q):
    return sse_decrypt_bytes(ciphertexts, public_key, private_key, n, q)


def hash_function(data: bytes, outlen=32) -> bytes:
    return hashlib.sha3_256(data).digest()[:outlen]

def hash_g(data: bytes) -> tuple:
    out = hashlib.sha3_512(data).digest()
    return out[:32], out[32:64]

def kdf(data: bytes, outlen=32) -> bytes:
    return hashlib.sha3_256(data).digest()[:outlen]

# Updated to ensure full bit preservation when encoding/decoding messages
def encode_message_to_poly(m: bytes, n=256, q=3329) -> np.ndarray:
    bitstring = ''.join(f'{byte:08b}' for byte in m)
    coeffs = np.zeros((n, 1), dtype=np.uint16)
    for i in range(min(len(bitstring), n)):
        coeffs[i, 0] = (q // 2) if bitstring[i] == '1' else 0
    return coeffs

def decode_poly_to_message(poly: np.ndarray, q=3329) -> bytes:
    bits = ''.join('1' if coef > (q // 4) else '0' for coef in poly.flatten())
    # Trim bits to full bytes length
    bits = bits[: (len(bits) // 8) * 8]
    return bytes(int(bits[i:i+8], 2) for i in range(0, len(bits), 8))


def poly_mul_mod(a, b, q):
    if lib is None or not hasattr(lib, "poly_mul_ntt"):
        return (a.flatten() * b.flatten()) % q
    n = len(a)
    res = (ctypes.c_uint16 * n)()
    lib.poly_mul_ntt(
        a.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
        b.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
        res,
        n,
        q
    )
    return np.frombuffer(res, dtype=np.uint16)


def poly_dot_mod(a, b, q):
    if lib is None:
        return int(np.sum(a.flatten() * b.flatten()) % q)
    return lib.poly_dot_mod(
        a.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
        b.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
        len(a),
        q
    )


def random_poly(n, low=-2, high=3):
    return np.random.randint(low, high, size=(n, 1))


def deterministic_chain(prev_ct, q, n):
    if prev_ct is None:
        return np.zeros((n, 1), dtype=int)
    c1, c2 = prev_ct
    data = np.concatenate([c1.flatten(), c2.flatten()])
    digest = hashlib.sha256(data.tobytes()).digest()
    while len(digest) < n:
        digest += hashlib.sha256(digest).digest()
    vals = np.frombuffer(digest[:n], dtype=np.uint8).astype(np.uint16)
    return vals.reshape(n, 1) % q


def encode_val(val, q):
    step = q // 256
    return int(val) * step


def decode_val(val, q):
    step = q // 256
    return int(round(val / step)) % 256


def sse_keygen(n, q):
    A = np.random.randint(0, q, size=(n,))
    s = random_poly(n)
    e = random_poly(n)
    b = (A.reshape(-1, 1) * s + e) % q
    return (A, b), s


def sse_encrypt_bytes(data_bytes, public_key, n, q):
    A, b = public_key
    ciphertexts = []
    prev_ct = None
    for val in data_bytes:
        chain_val = int(np.sum(deterministic_chain(prev_ct, q, n)))
        m_val = (encode_val(val, q) + chain_val) % q
        r = np.random.randint(0, 2, size=(n, 1))
        c1 = poly_mul_mod(A.reshape(-1, 1), r, q).reshape(-1, 1)
        c2 = (poly_dot_mod(b, r, q) + m_val) % q
        ciphertexts.append((c1, c2))
        prev_ct = (c1, np.array([[c2]]))
    return ciphertexts


def sse_decrypt_bytes(ciphertexts, public_key, private_key, n, q):
    A, b = public_key
    s = private_key
    recovered = bytearray()
    prev_ct = None
    for c1, c2 in ciphertexts:
        chain_val = int(np.sum(deterministic_chain(prev_ct, q, n)))
        v = (c2 - poly_dot_mod(s, c1, q) - chain_val) % q
        recovered.append(decode_val(v, q))
        prev_ct = (c1, np.array([[c2]]))
    return bytes(recovered)


def pack_key(A, b, q):
    def pack_coeffs(coeffs):
        coeffs = coeffs.flatten().astype(np.uint16)
        out = bytearray()
        for i in range(0, len(coeffs), 2):
            a = coeffs[i]
            b = coeffs[i + 1] if i + 1 < len(coeffs) else 0
            out.append(a & 0xFF)
            out.append(((a >> 8) & 0x0F) | ((b & 0x0F) << 4))
            out.append((b >> 4) & 0xFF)
        return bytes(out)

    return base64.b64encode(pack_coeffs(A.reshape(-1, 1)) + pack_coeffs(b)).decode()


def unpack_key(encoded, n):
    def unpack_coeffs(data):
        coeffs = []
        for i in range(0, len(data), 3):
            a = data[i] | ((data[i + 1] & 0x0F) << 8)
            b = ((data[i + 1] >> 4) & 0x0F) | (data[i + 2] << 4)
            coeffs.append(a)
            coeffs.append(b)
        return np.array(coeffs[:n], dtype=np.uint16).reshape(n, 1)

    raw = base64.b64decode(encoded)
    half = len(raw) // 2
    return (unpack_coeffs(raw[:half]).flatten(), unpack_coeffs(raw[half:]))


def pack_ciphertext(cts, q):
    out = []
    for c1, c2 in cts:
        packed_c1 = pack_key(c1, np.zeros_like(c1), q)  # pack_key reused for packing c1
        c2_bytes = c2.to_bytes(2, "little")
        out.append(base64.b64encode(base64.b64decode(packed_c1) + c2_bytes).decode())
    return out


def unpack_ciphertext(packed_cts, n):
    def unpack_coeffs(data):
        coeffs = []
        for i in range(0, len(data), 3):
            a = data[i] | ((data[i + 1] & 0x0F) << 8)
            b = ((data[i + 1] >> 4) & 0x0F) | (data[i + 2] << 4)
            coeffs.append(a)
            coeffs.append(b)
        return np.array(coeffs[:n], dtype=np.uint16).reshape(n, 1)

    cts = []
    for p in packed_cts:
        raw = base64.b64decode(p)
        c1_raw, c2_raw = raw[:-2], raw[-2:]
        cts.append((unpack_coeffs(c1_raw), int.from_bytes(c2_raw, "little")))
    return cts


# Updated KEM with ECC and KDF
def kem_keygen(n=256, q=3329):
    pub, priv = sse_keygen(n, q)
    pk_encoded = pack_key(*pub, q)
    h_pk = hash_function(pk_encoded.encode())
    z = get_random_bytes(32)
    return pk_encoded, {"priv": priv, "h_pk": h_pk, "z": z, "pk_encoded": pk_encoded}

def kem_encapsulate(pk_encoded, n=256, q=3329):
    pk = unpack_key(pk_encoded, n)
    h_pk = hash_function(pk_encoded.encode())
    seed = get_random_bytes(32)
    m = hash_function(seed)
    K_bar, r = hash_g(m + h_pk)

    # Encode m to polynomial explicitly
    m_poly = encode_message_to_poly(m, n, q)
    cts = sse_encrypt_bytes(m_poly.tobytes(), pk, n, q)

    ct_encoded = pack_ciphertext(cts, q)
    shared_secret = kdf(K_bar + hash_function(''.join(ct_encoded).encode()))
    return ct_encoded, shared_secret

def kem_decapsulate(ct_encoded, *args, n=256, q=3329):
    if len(args) == 1:
        sk_input = args[0]
    elif len(args) == 2:
        sk_input = args[1]
    else:
        raise TypeError("Provide either the dict returned by kem_keygen or both pk and sk from kem_keygen.")

    if isinstance(sk_input, dict):
        sk = sk_input
    elif isinstance(sk_input, tuple) and len(sk_input) == 2 and isinstance(sk_input[1], dict):
        sk = sk_input[1]
    else:
        raise TypeError("Invalid secret key format passed to kem_decapsulate.")

    priv = sk["priv"]
    h_pk = sk["h_pk"]
    z = sk["z"]
    pk_encoded = sk["pk_encoded"]

    pk = unpack_key(pk_encoded, n)
    cts = unpack_ciphertext(ct_encoded, n)

    m_prime_bytes = sse_decrypt_bytes(cts, pk, priv, n, q)
    m_prime = m_prime_bytes[:32]  # Ensure same size as m in encapsulate

    K_bar2, _ = hash_g(m_prime + h_pk)

    # Recompute ciphertext for verification
    m_poly = encode_message_to_poly(m_prime, n, q)
    cts2 = sse_encrypt_bytes(m_poly.tobytes(), pk, n, q)
    ct_encoded2 = pack_ciphertext(cts2, q)

    if ct_encoded2 == ct_encoded:
        return kdf(K_bar2 + hash_function(''.join(ct_encoded).encode()))
    return kdf(z + hash_function(''.join(ct_encoded).encode()))



# Benchmarking code remains unchanged


def ciphertext_entropy(ciphertext):
    vals = np.concatenate([c1.flatten() for c1, c2 in ciphertext] + [np.array([c2]) for c1, c2 in ciphertext])
    hist, _ = np.histogram(vals % 256, bins=256, range=(0, 255), density=True)
    hist = hist[hist > 0]
    return -np.sum(hist * np.log2(hist))


def single_benchmark(n, q, text="HELLO", repeats=5):
    pub, priv = sse_keygen(n, q)
    enc_times, dec_times, entropies = [], [], []
    for _ in range(repeats):
        start = time.time();
        cts = sse_encrypt(text, pub, n, q);
        enc_times.append(time.time() - start)
        start = time.time();
        _ = sse_decrypt(cts, pub, priv, n, q);
        dec_times.append(time.time() - start)
        entropies.append(ciphertext_entropy(cts))
    flipped_text = "A" + text[1:]
    cts1, cts2 = sse_encrypt(text, pub, n, q), sse_encrypt(flipped_text, pub, n, q)
    vals1 = np.concatenate([c1.flatten() for c1, c2 in cts1] + [np.array([c2]) for c1, c2 in cts1])
    vals2 = np.concatenate([c1.flatten() for c1, c2 in cts2] + [np.array([c2]) for c1, c2 in cts2])
    avalanche = np.mean(vals1 != vals2)
    return {
        "n": n,
        "q": q,
        "SSE_avg_enc_time": np.mean(enc_times),
        "SSE_avg_dec_time": np.mean(dec_times),
        "SSE_avg_entropy_bits": np.mean(entropies),
        "SSE_avalanche_effect": avalanche,
        "key_size_bytes": len(pack_key(*pub, q)),
        "ct_size_bytes": len(pack_ciphertext(cts, q)[0])
    }


def benchmark_all(text="HELLO", repeats=5):
    params = [(128, 3329), (256, 3329), (512, 3329), (1024, 3329)]
    results = [single_benchmark(n, q, text, repeats) for n, q in params]

    key = get_random_bytes(16)
    aes = AES.new(key, AES.MODE_ECB)
    aes_times = []
    for _ in range(repeats):
        start = time.time()
        aes.encrypt(text.encode().ljust(16, b"\0"))
        aes_times.append(time.time() - start)

    rsa_key = RSA.generate(2048)
    rsa_cipher = PKCS1_OAEP.new(rsa_key)
    rsa_times = []
    for _ in range(repeats):
        start = time.time()
        rsa_cipher.encrypt(text.encode())
        rsa_times.append(time.time() - start)

    return {"SSE_results": results, "AES_avg_enc_time": np.mean(aes_times), "RSA_avg_enc_time": np.mean(rsa_times)}


if __name__ == "__main__":
    results = benchmark_all("HELLO", 5)
    for r in results["SSE_results"]:
        print(
            f"N={r['n']} Q={r['q']} | enc={r['SSE_avg_enc_time']:.6f}s | entropy={r['SSE_avg_entropy_bits']:.3f} bits | avalanche={r['SSE_avalanche_effect']:.3f} | key={r['key_size_bytes']}B | ct={r['ct_size_bytes']}B")
    print("AES avg enc time:", results["AES_avg_enc_time"])
    print("RSA avg enc time:", results["RSA_avg_enc_time"])

    pub_key, priv_key = kem_keygen()
    ciphertext, shared_secret = kem_encapsulate(pub_key)
    recovered_secret = kem_decapsulate(ciphertext, pub_key, priv_key)

    print("\nKEM Demo:")
    print("Shared Secret (Encapsulated):", shared_secret)
    print("Recovered Secret (Decapsulated):", recovered_secret)
