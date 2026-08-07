// 256-bit Montgomery field arithmetic for HIP (AMD GPUs).
//
// Drop-in replacement for mont_t.cuh, compiled only under __HIPCC__.
// Uses __builtin_add_overflow / __builtin_sub_overflow for carry chains,
// which LLVM lowers to optimal RDNA3 instructions:
//   v_add_co_u32 / v_add_co_ci_u32_e32 (chained carry add)
//   v_sub_co_u32 / v_sub_co_ci_u32_e32 (chained borrow sub)
// The inner REDC loop uses uint64_t multiply-accumulate patterns
// that LLVM lowers to v_mad_u64_u32 (32×32→64 with accumulate).
//
// Template parameters match mont_t.cuh exactly:
//   N   – bit-length of the modulus (254 for BN254)
//   MOD – modulus as an array of n = ceil(N/32) uint32_t limbs
//   M0  – Montgomery factor: -(MOD^{-1}) mod 2^32
//   RR  – (2^{2*n*32}) mod MOD
//   ONE – (2^{n*32}) mod MOD  (Montgomery form of 1)

#if defined(__HIPCC__) && !defined(__MONT_T_HIP__)
#define __MONT_T_HIP__

#include <cstddef>
#include <cstdint>

#define inline __device__ __forceinline__

template <
    const size_t N,
    const uint32_t MOD[(N + 31) / 32],
    const uint32_t &M0,
    const uint32_t RR[(N + 31) / 32],
    const uint32_t ONE[(N + 31) / 32]>
class __align__(((N + 63) / 64) & 1 ? 8 : 16) mont_t {
public:
    static const size_t nbits = N;
    static const size_t nbytes = (N + 7) / 8;

private:
    static const size_t n = (N + 31) / 32;
    uint32_t even[n];

    // ── helpers ────────────────────────────────────────────────────

    static inline bool is_ge_mod(const uint32_t *a) {
        for (size_t i = n; i--;) {
            if (a[i] > MOD[i]) return true;
            if (a[i] < MOD[i]) return false;
        }
        return true; // equal
    }

    static inline bool is_gt(const uint32_t *r, const uint32_t *l) {
        for (size_t i = n; i--;) {
            if (r[i] > l[i]) return true;
            if (r[i] < l[i]) return false;
        }
        return false;
    }

    static inline bool is_equal(const uint32_t *r, const uint32_t *l) {
        for (size_t i = 0; i < n; i++)
            if (r[i] != l[i]) return false;
        return true;
    }

    static inline uint32_t add_limbs(uint32_t *r, const uint32_t *a,
                                     const uint32_t *b) {
        bool carry = false;
        for (size_t i = 0; i < n; i++) {
            uint32_t sum;
            bool c1 = __builtin_add_overflow(a[i], b[i], &sum);
            bool c2 = __builtin_add_overflow(sum, (uint32_t)carry, &r[i]);
            carry = c1 | c2;
        }
        return carry ? 1 : 0;
    }

    static inline uint32_t sub_limbs(uint32_t *r, const uint32_t *a,
                                     const uint32_t *b) {
        bool borrow = false;
        for (size_t i = 0; i < n; i++) {
            uint32_t diff;
            bool b1 = __builtin_sub_overflow(a[i], b[i], &diff);
            bool b2 = __builtin_sub_overflow(diff, (uint32_t)borrow, &r[i]);
            borrow = b1 | b2;
        }
        return borrow ? 1 : 0;
    }

    // Montgomery REDC: given 2n-limb product T, compute T * R^{-1} mod MOD
    static inline void redc(uint32_t *result, uint32_t *T) {
        for (size_t i = 0; i < n; i++) {
            uint32_t m = T[i] * M0;
            uint64_t carry = 0;
            for (size_t j = 0; j < n; j++) {
                carry += (uint64_t)m * MOD[j] + T[i + j];
                T[i + j] = (uint32_t)carry;
                carry >>= 32;
            }
            for (size_t j = i + n; carry && j < 2 * n; j++) {
                uint32_t sum;
                bool c = __builtin_add_overflow(T[j], (uint32_t)carry, &sum);
                T[j] = sum;
                carry = c ? 1 : 0;
            }
        }
        for (size_t i = 0; i < n; i++)
            result[i] = T[i + n];
        if (is_ge_mod(result)) {
            sub_limbs(result, result, MOD);
        }
    }

public:
    inline operator const uint32_t *() const { return even; }
    inline operator uint32_t *() { return even; }

    inline uint32_t &operator[](size_t i) { return even[i]; }
    inline const uint32_t &operator[](size_t i) const { return even[i]; }

    __host__ __device__ inline mont_t() {}
    __host__ __device__ inline mont_t(const uint32_t *p) {
        for (size_t i = 0; i < n; i++) even[i] = p[i];
    }

    __host__ __device__ inline void store(uint32_t *p) const {
        for (size_t i = 0; i < n; i++) p[i] = even[i];
    }

    // ── addition ──

    inline mont_t &operator+=(const mont_t &b) {
        add_limbs(even, even, b.even);
        if (is_ge_mod(even)) sub_limbs(even, even, MOD);
        return *this;
    }
    friend inline mont_t operator+(mont_t a, const mont_t &b) { return a += b; }

    inline uint32_t add_with_carry(const mont_t &b) {
        return add_limbs(even, even, b.even);
    }

    // ── left shift ──

    inline mont_t &operator<<=(unsigned l) {
        while (l--) {
            uint32_t carry = 0;
            for (size_t i = 0; i < n; i++) {
                uint32_t tmp = even[i];
                even[i] = (tmp << 1) | carry;
                carry = tmp >> 31;
            }
            if (carry || is_ge_mod(even))
                sub_limbs(even, even, MOD);
        }
        return *this;
    }
    friend inline mont_t operator<<(mont_t a, unsigned l) { return a <<= l; }

    // ── right shift ──

    inline mont_t &operator>>=(unsigned r) {
        while (r--) {
            if (even[0] & 1) add_limbs(even, even, MOD);
            uint32_t carry = 0;
            for (size_t i = n; i--;) {
                uint32_t tmp = even[i];
                even[i] = (tmp >> 1) | (carry << 31);
                carry = tmp & 1;
            }
        }
        return *this;
    }
    friend inline mont_t operator>>(mont_t a, unsigned r) { return a >>= r; }

    inline void div_by_2(uint32_t carry = 0) {
        for (size_t i = n; i--;) {
            uint32_t tmp = even[i];
            even[i] = (tmp >> 1) | (carry << 31);
            carry = tmp & 1;
        }
    }

    // ── subtraction ──

    inline mont_t &operator-=(const mont_t &b) {
        uint32_t borrow = sub_limbs(even, even, b.even);
        if (borrow) add_limbs(even, even, MOD);
        return *this;
    }
    friend inline mont_t operator-(mont_t a, const mont_t &b) { return a -= b; }

    inline mont_t &cneg(bool flag) {
        if (flag) {
            bool is_z = true;
            for (size_t i = 0; i < n; i++) is_z &= (even[i] == 0);
            if (!is_z) {
                sub_limbs(even, MOD, even);
            }
        }
        return *this;
    }
    friend inline mont_t cneg(mont_t a, bool flag) { return a.cneg(flag); }
    friend inline mont_t cneg(const mont_t &a, bool flag) {
        mont_t ret = a; ret.cneg(flag); return ret;
    }
    inline mont_t operator-() const { return cneg(*this, true); }

    // ── multiplication (Montgomery) ──

    friend inline mont_t operator*(const mont_t &a, const mont_t &b) {
        uint32_t T[2 * n] = {};
        // schoolbook multiply
        for (size_t i = 0; i < n; i++) {
            uint64_t carry = 0;
            for (size_t j = 0; j < n; j++) {
                carry += (uint64_t)a.even[i] * b.even[j] + T[i + j];
                T[i + j] = (uint32_t)carry;
                carry >>= 32;
            }
            T[i + n] = (uint32_t)carry;
        }
        mont_t ret;
        redc(ret.even, T);
        return ret;
    }
    inline mont_t &operator*=(const mont_t &a) { return *this = *this * a; }

    // ── power / square ──

    inline mont_t &operator^=(unsigned p) {
        if (p < 2) {
            if (p == 0) { for (size_t i = 0; i < n; i++) even[i] = ONE[i]; }
            return *this;
        }
        mont_t sqr_val = *this;
        if ((p & 1) == 0) {
            // find highest bit
            do {
                sqr_val *= sqr_val;
                p >>= 1;
            } while ((p & 1) == 0);
            *this = sqr_val;
        }
        for (p >>= 1; p; p >>= 1) {
            sqr_val *= sqr_val;
            if (p & 1)
                *this *= sqr_val;
        }
        return *this;
    }
    friend inline mont_t operator^(mont_t a, unsigned p) { return a ^= p; }
    inline mont_t operator()(unsigned p) { return *this ^ p; }
    friend inline mont_t sqr(const mont_t &a) { return a ^ 2; }

    // ── comparison ──

    friend inline bool operator==(const mont_t &a, const mont_t &b) {
        return is_equal(a.even, b.even);
    }
    friend inline bool operator!=(const mont_t &a, const mont_t &b) { return !(a == b); }
    friend inline bool operator>(const mont_t &a, const mont_t &b) { return is_gt(a.even, b.even); }
    friend inline bool operator>=(const mont_t &a, const mont_t &b) {
        return !is_gt(b.even, a.even);
    }

    // ── to/from Montgomery form ──

    inline void to() {
        uint32_t T[2 * n] = {};
        for (size_t i = 0; i < n; i++) {
            uint64_t carry = 0;
            for (size_t j = 0; j < n; j++) {
                carry += (uint64_t)even[i] * RR[j] + T[i + j];
                T[i + j] = (uint32_t)carry;
                carry >>= 32;
            }
            T[i + n] = (uint32_t)carry;
        }
        redc(even, T);
    }

    inline void from() {
        uint32_t T[2 * n] = {};
        for (size_t i = 0; i < n; i++) T[i] = even[i];
        redc(even, T);
    }

    // ── predicates ──

    inline bool is_zero() const {
        uint32_t acc = 0;
        for (size_t i = 0; i < n; i++) acc |= even[i];
        return acc == 0;
    }

    inline void zero() {
        for (size_t i = 0; i < n; i++) even[i] = 0;
    }

    static inline const mont_t &one() {
        return *reinterpret_cast<const mont_t *>(ONE);
    }
    static inline const mont_t &rr() {
        return *reinterpret_cast<const mont_t *>(RR);
    }
    static inline const mont_t &modulus() {
        return *reinterpret_cast<const mont_t *>(MOD);
    }
    inline bool is_one() const { return is_equal(even, ONE); }

    inline bool is_odd() { return even[0] & 0x1; }
    inline bool is_even() { return !is_odd(); }

    // ── conditional select / zero ──

    friend inline mont_t czero(const mont_t &a, int set_z) {
        mont_t ret;
        for (size_t i = 0; i < n; i++)
            ret.even[i] = set_z ? 0 : a.even[i];
        return ret;
    }

    static inline mont_t csel(const mont_t &a, const mont_t &b, int sel_a) {
        mont_t ret;
        for (size_t i = 0; i < n; i++)
            ret.even[i] = sel_a ? a.even[i] : b.even[i];
        return ret;
    }

    // ── inverse (binary GCD — Botrel & El Housni, TCHES 2023) ──
    // ~512 shifts+adds vs 254 multiplications (Fermat). Both vendors benefit
    // from reduced register pressure and elimination of the expensive mul chain.
    //
    // Invariant: u * RR ≡ b * (a*R) (mod p)
    // At termination u=1: b ≡ R/a ≡ a^{-1}*R (mod p) — Montgomery inverse.

    inline bool is_limb_one() const {
        if (even[0] != 1) return false;
        for (size_t i = 1; i < n; i++)
            if (even[i] != 0) return false;
        return true;
    }

    inline mont_t inverse() {
        if (is_zero())
            __builtin_trap();

        mont_t u(*this);           // u = a*R (the stored value)
        mont_t v(MOD);             // v = p
        mont_t b(RR);              // b = R^2 mod p
        mont_t c;
        c.zero();                  // c = 0

        while (!u.is_limb_one() && !v.is_limb_one()) {
            while (u.is_even()) {
                u.div_by_2();
                uint32_t carry = 0;
                if (b.is_odd()) carry = add_limbs(b.even, b.even, MOD);
                b.div_by_2(carry);
            }
            while (v.is_even()) {
                v.div_by_2();
                uint32_t carry = 0;
                if (c.is_odd()) carry = add_limbs(c.even, c.even, MOD);
                c.div_by_2(carry);
            }
            if (u >= v) {
                sub_limbs(u.even, u.even, v.even);
                b -= c;
            } else {
                sub_limbs(v.even, v.even, u.even);
                c -= b;
            }
        }

        return u.is_limb_one() ? b : c;
    }

    inline mont_t reciprocal() const {
        mont_t tmp = *this;
        return tmp.inverse();
    }
    friend inline mont_t operator/(int one, mont_t a) {
        return a.reciprocal();
    }
    friend inline mont_t operator/(mont_t a, mont_t b) {
        return a * b.reciprocal();
    }
    inline mont_t &operator/=(const mont_t a) {
        return *this *= a.reciprocal();
    }

    // ── mul_by_1: Montgomery reduction of (even, 0..0) ──

    inline void mul_by_1() {
        from();
    }

    inline bool equal_to_digital_one() {
        mont_t tmp = *this;
        tmp.from();
        if (tmp.even[0] != 1) return false;
        for (size_t i = 1; i < n; i++)
            if (tmp.even[i] != 0) return false;
        return true;
    }

    // ── warp shuffle (for reductions) ──

    inline void shfl_bfly(uint32_t laneMask) {
        uint32_t idx = (threadIdx.x ^ laneMask) << 2;
        for (size_t i = 0; i < n; i++)
            even[i] = __builtin_amdgcn_ds_bpermute(idx, even[i]);
    }
};

#undef inline
#endif // __MONT_T_HIP__
