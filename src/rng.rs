pub struct Rng {
    s: [u64; 4],
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        let mut r = Rng { s: [0x9E3779B97F4A7C15, 0xBF58476D1CE4E5B9, 0x94D049BB133111EB, seed.wrapping_add(0xD1B54A32D192ED03)] };
        if seed == 0 {
            r.s[0] = 0x9E3779B97F4A7C15;
            r.s[1] = 0xBF58476D1CE4E5B9;
            r.s[2] = 0x94D049BB133111EB;
            r.s[3] = 0xD1B54A32D192ED03;
        }
        r
    }

    pub fn next_u64(&mut self) -> u64 {
        let r = self.s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        r
    }

    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    pub fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.next_f32()
    }

    pub fn gauss(&mut self) -> f32 {
        let u1 = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        let u2 = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        let r = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        r as f32
    }

    pub fn shuffle(&mut self, items: &mut [usize]) {
        for i in (1..items.len()).rev() {
            let j = (self.next_u64() as usize) % (i + 1);
            items.swap(i, j);
        }
    }
}
