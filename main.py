import numpy as np
import hashlib, base64, os, time, ctypes
from Crypto.Cipher import AES, PKCS1_OAEP
from Crypto.PublicKey import RSA
from Crypto.Random import get_random_bytes

ALPHABET_B32 = "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567"

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

def pack_coeffs(coeffs, q):
    coeffs = coeffs.flatten().astype(np.uint16)
    out = bytearray()
    for i in range(0, len(coeffs), 2):
        a = coeffs[i]
        b = coeffs[i+1] if i+1 < len(coeffs) else 0
        out.append(a & 0xFF)
        out.append(((a >> 8) & 0x0F) | ((b & 0x0F) << 4))
        out.append((b >> 4) & 0xFF)
    return bytes(out)

def unpack_coeffs(data, n):
    coeffs = []
    for i in range(0, len(data), 3):
        a = data[i] | ((data[i+1] & 0x0F) << 8)
        b = ((data[i+1] >> 4) & 0x0F) | (data[i+2] << 4)
        coeffs.append(a)
        coeffs.append(b)
    return np.array(coeffs[:n], dtype=np.uint16).reshape(n, 1)

def sse_keygen(n, q):
    A = np.random.randint(0, q, size=(n,))
    s = random_poly(n)
    e = random_poly(n)
    b = (A.reshape(-1, 1) * s + e) % q
    return (A, b), s

def sse_encrypt(data, public_key, n, q):
    if isinstance(data, str):
        data = data.encode()
    A, b = public_key
    ciphertexts = []
    prev_ct = None
    for val in data:
        chain_val = int(np.sum(deterministic_chain(prev_ct, q, n)))
        m_val = (encode_val(val, q) + chain_val) % q
        r = np.random.randint(0, 2, size=(n, 1))
        c1 = poly_mul_mod(A.reshape(-1, 1), r, q).reshape(-1, 1)
        c2 = (poly_dot_mod(b, r, q) + m_val) % q
        ciphertexts.append((c1, c2))
        prev_ct = (c1, np.array([[c2]]))
    return ciphertexts

def sse_decrypt(ciphertexts, public_key, private_key, n, q):
    A, b = public_key
    s = private_key
    recovered_bytes = bytearray()
    prev_ct = None
    for c1, c2 in ciphertexts:
        chain_val = int(np.sum(deterministic_chain(prev_ct, q, n)))
        v = (c2 - poly_dot_mod(s, c1, q) - chain_val) % q
        recovered_bytes.append(decode_val(v, q))
        prev_ct = (c1, np.array([[c2]]))
    return bytes(recovered_bytes)

def pack_key(A, b, q):
    return base64.b64encode(pack_coeffs(A.reshape(-1, 1), q) + pack_coeffs(b, q)).decode()

def unpack_key(encoded, n):
    raw = base64.b64decode(encoded)
    size_half = len(raw) // 2
    return (unpack_coeffs(raw[:size_half], n).flatten(), unpack_coeffs(raw[size_half:], n))

def pack_ciphertext(cts, q):
    out = []
    for c1, c2 in cts:
        packed_c1 = pack_coeffs(c1, q)
        c2_bytes = c2.to_bytes(2, "little")
        out.append(base64.b64encode(packed_c1 + c2_bytes).decode())
    return out

def unpack_ciphertext(packed_cts, n):
    cts = []
    for p in packed_cts:
        raw = base64.b64decode(p)
        cts.append((unpack_coeffs(raw[:-2], n), int.from_bytes(raw[-2:], "little")))
    return cts

def kem_keygen(n=256, q=3329):
    pub, priv = sse_keygen(n, q)
    return pack_key(*pub, q), priv

def kem_encapsulate(pub_key_encoded, n=256, q=3329):
    pub = unpack_key(pub_key_encoded, n)
    r = get_random_bytes(16)
    cts = sse_encrypt(r, pub, n, q)
    shared_secret = hashlib.sha256(r).hexdigest()
    return pack_ciphertext(cts, q), shared_secret

def kem_decapsulate(cts_encoded, pub_key_encoded, priv, n=256, q=3329):
    pub = unpack_key(pub_key_encoded, n)
    cts = unpack_ciphertext(cts_encoded, n)
    decrypted_bytes = sse_decrypt(cts, pub, priv, n, q)
    r = decrypted_bytes[:16]
    return hashlib.sha256(r).hexdigest()

def ciphertext_entropy(ciphertext):
    vals = np.concatenate([c1.flatten() for c1, c2 in ciphertext] + [np.array([c2]) for c1, c2 in ciphertext])
    hist, _ = np.histogram(vals % 256, bins=256, range=(0, 255), density=True)
    hist = hist[hist > 0]
    return -np.sum(hist * np.log2(hist))

def single_benchmark(n, q, text="HELLO", repeats=5):
    pub, priv = sse_keygen(n, q)
    enc_times, dec_times, entropies = [], [], []
    for _ in range(repeats):
        start = time.time(); cts = sse_encrypt(text, pub, n, q); enc_times.append(time.time() - start)
        start = time.time(); _ = sse_decrypt(cts, pub, priv, n, q); dec_times.append(time.time() - start)
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
        print(f"N={r['n']} Q={r['q']} | enc={r['SSE_avg_enc_time']:.6f}s | entropy={r['SSE_avg_entropy_bits']:.3f} bits | avalanche={r['SSE_avalanche_effect']:.3f} | key={r['key_size_bytes']}B | ct={r['ct_size_bytes']}B")
    print("AES avg enc time:", results["AES_avg_enc_time"])
    print("RSA avg enc time:", results["RSA_avg_enc_time"])

    pub_key, priv_key = kem_keygen()
    ciphertext, shared_secret = kem_encapsulate(pub_key)
    recovered_secret = kem_decapsulate(ciphertext, pub_key, priv_key)

    print("\nKEM Demo:")
    print("Shared Secret (Encapsulated):", shared_secret)
    print("Recovered Secret (Decapsulated):", recovered_secret)
