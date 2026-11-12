//! Element van der Waals radius tables (Å).
//!
//! # Provenance
//!
//! **Bondi** (default set): A. Bondi, *J. Phys. Chem.* 68, 441 (1964),
//! completed for the 16 main-group elements Bondi did not cover (Be, B, Al,
//! Ca, Ge, Rb, Sr, Sb, Cs, Ba, Bi, Po, At, Rn, Fr, Ra) with M. Mantina et al.,
//! *J. Phys. Chem. A* 113, 5806 (2009), Table 12. Main-group values are taken
//! from Mantina's Table 12 (which reproduces Bondi's), except hydrogen, where
//! Mantina substitutes Rowland & Taylor's 1.10 Å and this set keeps Bondi's
//! 1.20 Å. Transition metals and U follow the "Bondi" column of Alvarez
//! (2013) Table 1. Where sources disagree (Si: 2.10 vs 2.22, Te: 2.06 vs
//! 2.00) Mantina's Table 12 is used. Elements absent from both papers (e.g.
//! Fe, Mn, Co) have no radius here and fall back to the configured default.
//!
//! **Alvarez**: S. Alvarez, *Dalton Trans.* 42, 8617 (2013), Table 1
//! (`r_vdW` column). Values given in square brackets in the paper (rough
//! estimates from very small datasets) are included and marked.

/// `(Z, radius)` pairs of the Bondi set (see module docs).
pub(super) const BONDI: &[(u8, f64)] = &[
    (1, 1.20),  // H  (Bondi; Mantina uses 1.10)
    (2, 1.40),  // He
    (3, 1.81),  // Li
    (4, 1.53),  // Be (Mantina)
    (5, 1.92),  // B  (Mantina)
    (6, 1.70),  // C
    (7, 1.55),  // N
    (8, 1.52),  // O
    (9, 1.47),  // F
    (10, 1.54), // Ne
    (11, 2.27), // Na
    (12, 1.73), // Mg
    (13, 1.84), // Al (Mantina)
    (14, 2.10), // Si
    (15, 1.80), // P
    (16, 1.80), // S
    (17, 1.75), // Cl
    (18, 1.88), // Ar
    (19, 2.75), // K
    (20, 2.31), // Ca (Mantina)
    (28, 1.63), // Ni
    (29, 1.40), // Cu
    (30, 1.39), // Zn
    (31, 1.87), // Ga
    (32, 2.11), // Ge (Mantina)
    (33, 1.85), // As
    (34, 1.90), // Se
    (35, 1.83), // Br
    (36, 2.02), // Kr
    (37, 3.03), // Rb (Mantina)
    (38, 2.49), // Sr (Mantina)
    (46, 1.63), // Pd
    (47, 1.72), // Ag
    (48, 1.62), // Cd
    (49, 1.93), // In
    (50, 2.17), // Sn
    (51, 2.06), // Sb (Mantina)
    (52, 2.06), // Te
    (53, 1.98), // I
    (54, 2.16), // Xe
    (55, 3.43), // Cs (Mantina)
    (56, 2.68), // Ba (Mantina)
    (78, 1.72), // Pt
    (79, 1.66), // Au
    (80, 1.70), // Hg
    (81, 1.96), // Tl
    (82, 2.02), // Pb
    (83, 2.07), // Bi (Mantina)
    (84, 1.97), // Po (Mantina)
    (85, 2.02), // At (Mantina)
    (86, 2.20), // Rn (Mantina)
    (87, 3.48), // Fr (Mantina)
    (88, 2.83), // Ra (Mantina)
    (92, 1.86), // U
];

/// `(Z, radius)` pairs of the Alvarez (2013) set.
pub(super) const ALVAREZ: &[(u8, f64)] = &[
    (1, 1.20),  // H
    (2, 1.43),  // He (rough estimate)
    (3, 2.12),  // Li
    (4, 1.98),  // Be
    (5, 1.91),  // B
    (6, 1.77),  // C
    (7, 1.66),  // N
    (8, 1.50),  // O
    (9, 1.46),  // F
    (10, 1.58), // Ne (rough estimate)
    (11, 2.50), // Na
    (12, 2.51), // Mg
    (13, 2.25), // Al
    (14, 2.19), // Si
    (15, 1.90), // P
    (16, 1.89), // S
    (17, 1.82), // Cl
    (18, 1.83), // Ar
    (19, 2.73), // K
    (20, 2.62), // Ca
    (21, 2.58), // Sc
    (22, 2.46), // Ti
    (23, 2.42), // V
    (24, 2.45), // Cr
    (25, 2.45), // Mn
    (26, 2.44), // Fe
    (27, 2.40), // Co
    (28, 2.40), // Ni
    (29, 2.38), // Cu
    (30, 2.39), // Zn
    (31, 2.32), // Ga
    (32, 2.29), // Ge
    (33, 1.88), // As
    (34, 1.82), // Se
    (35, 1.86), // Br
    (36, 2.25), // Kr
    (37, 3.21), // Rb
    (38, 2.84), // Sr
    (39, 2.75), // Y
    (40, 2.52), // Zr
    (41, 2.56), // Nb
    (42, 2.45), // Mo
    (43, 2.44), // Tc
    (44, 2.46), // Ru
    (45, 2.44), // Rh
    (46, 2.15), // Pd
    (47, 2.53), // Ag
    (48, 2.49), // Cd
    (49, 2.43), // In
    (50, 2.42), // Sn
    (51, 2.47), // Sb
    (52, 1.99), // Te
    (53, 2.04), // I
    (54, 2.06), // Xe
    (55, 3.48), // Cs
    (56, 3.03), // Ba
    (57, 2.98), // La
    (58, 2.88), // Ce
    (59, 2.92), // Pr
    (60, 2.95), // Nd
    (62, 2.90), // Sm
    (63, 2.87), // Eu
    (64, 2.83), // Gd
    (65, 2.79), // Tb
    (66, 2.87), // Dy
    (67, 2.81), // Ho
    (68, 2.83), // Er
    (69, 2.79), // Tm
    (70, 2.80), // Yb
    (71, 2.74), // Lu
    (72, 2.63), // Hf
    (73, 2.53), // Ta
    (74, 2.57), // W
    (75, 2.49), // Re
    (76, 2.48), // Os
    (77, 2.41), // Ir
    (78, 2.29), // Pt
    (79, 2.32), // Au
    (80, 2.45), // Hg
    (81, 2.47), // Tl
    (82, 2.60), // Pb
    (83, 2.54), // Bi
    (89, 2.80), // Ac (rough estimate)
    (90, 2.93), // Th
    (91, 2.88), // Pa (rough estimate)
    (92, 2.71), // U
    (93, 2.82), // Np
    (94, 2.81), // Pu
    (95, 2.83), // Am
    (96, 3.05), // Cm (rough estimate)
    (97, 3.40), // Bk (rough estimate)
    (98, 3.05), // Cf (rough estimate)
    (99, 2.70), // Es (rough estimate)
];

/// Dense lookup array indexed by atomic number; 0.0 means "no value".
pub(super) const fn dense(pairs: &[(u8, f64)]) -> [f64; 119] {
    let mut table = [0.0; 119];
    let mut i = 0;
    while i < pairs.len() {
        table[pairs[i].0 as usize] = pairs[i].1;
        i += 1;
    }
    table
}

pub(super) const BONDI_DENSE: [f64; 119] = dense(BONDI);
pub(super) const ALVAREZ_DENSE: [f64; 119] = dense(ALVAREZ);

#[cfg(test)]
mod tests {
    use super::*;

    /// Mantina et al. (2009) Table 12 as transcribed by QCElemental
    /// (`mantina_2009_vanderwaals_radii.py`), an independent machine-readable
    /// source.
    const MANTINA_TABLE_12: &[(u8, f64)] = &[
        (1, 1.10),
        (2, 1.40),
        (3, 1.81),
        (4, 1.53),
        (5, 1.92),
        (6, 1.70),
        (7, 1.55),
        (8, 1.52),
        (9, 1.47),
        (10, 1.54),
        (11, 2.27),
        (12, 1.73),
        (13, 1.84),
        (14, 2.10),
        (15, 1.80),
        (16, 1.80),
        (17, 1.75),
        (18, 1.88),
        (19, 2.75),
        (20, 2.31),
        (31, 1.87),
        (32, 2.11),
        (33, 1.85),
        (34, 1.90),
        (35, 1.83),
        (36, 2.02),
        (37, 3.03),
        (38, 2.49),
        (49, 1.93),
        (50, 2.17),
        (51, 2.06),
        (52, 2.06),
        (53, 1.98),
        (54, 2.16),
        (55, 3.43),
        (56, 2.68),
        (81, 1.96),
        (82, 2.02),
        (83, 2.07),
        (84, 1.97),
        (85, 2.02),
        (86, 2.20),
        (87, 3.48),
        (88, 2.83),
    ];

    /// Bondi's transition-metal and U values as quoted in Alvarez (2013),
    /// Table 1, "Bondi" column.
    const BONDI_PER_ALVAREZ: &[(u8, f64)] = &[
        (28, 1.63),
        (29, 1.40),
        (30, 1.39),
        (46, 1.63),
        (47, 1.72),
        (48, 1.62),
        (78, 1.72),
        (79, 1.66),
        (80, 1.70),
        (92, 1.86),
    ];

    #[test]
    fn bondi_matches_sources() {
        for &(z, r) in MANTINA_TABLE_12 {
            let expected = if z == 1 { 1.20 } else { r };
            assert_eq!(BONDI_DENSE[z as usize], expected, "Z = {z}");
        }
        for &(z, r) in BONDI_PER_ALVAREZ {
            assert_eq!(BONDI_DENSE[z as usize], r, "Z = {z}");
        }
        assert_eq!(
            BONDI.len(),
            MANTINA_TABLE_12.len() + BONDI_PER_ALVAREZ.len()
        );
    }

    #[test]
    fn tables_are_sorted_and_plausible() {
        for table in [BONDI, ALVAREZ] {
            assert!(table.windows(2).all(|w| w[0].0 < w[1].0));
            assert!(
                table
                    .iter()
                    .all(|&(z, r)| (1..=118).contains(&z) && (1.0..3.6).contains(&r))
            );
        }
        assert_eq!(ALVAREZ.len(), 93);
        // Spot checks against Alvarez (2013) Table 1.
        assert_eq!(ALVAREZ_DENSE[6], 1.77);
        assert_eq!(ALVAREZ_DENSE[8], 1.50);
        assert_eq!(ALVAREZ_DENSE[26], 2.44);
        assert_eq!(ALVAREZ_DENSE[61], 0.0); // Pm: no value
    }
}
