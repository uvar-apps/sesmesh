//! User-defined radius rules (Spec FR-RAD-3).

use rustc_hash::FxHashMap;

use super::BuiltinRadii;
use crate::error::{Error, Result};
use crate::structure::Element;

/// Custom radius rules layered over an optional built-in set.
///
/// Lookup order for an atom: `RES:ATOM`, then `*:ATOM`, then `RES:*`, then
/// the element rule, then the base set, then the rules' fallback (or the
/// global unknown radius). Residue and atom names are case-insensitive.
///
/// The TOML form (feature `radii-file`):
///
/// ```toml
/// base = "bondi"        # bondi | alvarez | protor | none (default: bondi)
/// fallback = 1.8        # optional
///
/// [elements]
/// Fe = 2.0
///
/// [atoms]               # "RESIDUE:ATOM", either part may be "*"
/// "HEM:FE" = 1.6
/// "*:OXT" = 1.52
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct RadiusRules {
    base: Option<BuiltinRadii>,
    fallback: Option<f64>,
    elements: FxHashMap<Element, f64>,
    atoms: FxHashMap<(String, String), f64>,
}

impl Default for RadiusRules {
    fn default() -> Self {
        Self::new(Some(BuiltinRadii::Bondi))
    }
}

fn check_radius(r: f64, what: &str) -> Result<f64> {
    if r.is_finite() && r >= 0.0 {
        Ok(r)
    } else {
        Err(Error::InvalidInput(format!(
            "radius for {what} must be a finite non-negative number, got {r}"
        )))
    }
}

impl RadiusRules {
    /// Empty rules over `base` (`None`: no built-in set).
    pub fn new(base: Option<BuiltinRadii>) -> Self {
        Self {
            base,
            fallback: None,
            elements: FxHashMap::default(),
            atoms: FxHashMap::default(),
        }
    }

    /// The built-in set consulted after the rules.
    pub fn base(&self) -> Option<BuiltinRadii> {
        self.base
    }

    /// Sets the radius used when nothing else matches.
    ///
    /// # Errors
    /// [`Error::InvalidInput`] for negative or non-finite radii.
    pub fn set_fallback(&mut self, radius: f64) -> Result<()> {
        self.fallback = Some(check_radius(radius, "fallback")?);
        Ok(())
    }

    /// The fallback radius, if set.
    pub fn fallback(&self) -> Option<f64> {
        self.fallback
    }

    /// Sets the radius of every atom of `element` not matched by an atom rule.
    ///
    /// # Errors
    /// [`Error::InvalidInput`] for negative or non-finite radii.
    pub fn set_element(&mut self, element: Element, radius: f64) -> Result<()> {
        self.elements
            .insert(element, check_radius(radius, element.symbol())?);
        Ok(())
    }

    /// Sets the radius of atom `atom` in residue `residue`; either may be `*`.
    ///
    /// # Errors
    /// [`Error::InvalidInput`] for negative or non-finite radii or empty names.
    pub fn set_atom(&mut self, residue: &str, atom: &str, radius: f64) -> Result<()> {
        let (residue, atom) = (residue.trim(), atom.trim());
        if residue.is_empty() || atom.is_empty() {
            return Err(Error::InvalidInput(format!(
                "atom rule '{residue}:{atom}' needs a residue and an atom name (or '*')"
            )));
        }
        let r = check_radius(radius, &format!("{residue}:{atom}"))?;
        self.atoms
            .insert((residue.to_ascii_uppercase(), atom.to_ascii_uppercase()), r);
        Ok(())
    }

    /// The radius given by the rules themselves (not the base set or the
    /// fallback).
    pub(crate) fn lookup_rules(
        &self,
        residue: &str,
        atom: &str,
        element: Option<Element>,
    ) -> Option<f64> {
        if !self.atoms.is_empty() {
            let (res, at) = (residue.to_ascii_uppercase(), atom.to_ascii_uppercase());
            for key in [
                (res.as_str(), at.as_str()),
                ("*", at.as_str()),
                (res.as_str(), "*"),
            ] {
                if let Some(&r) = self.atoms.get(&(key.0.to_owned(), key.1.to_owned())) {
                    return Some(r);
                }
            }
        }
        element.and_then(|e| self.elements.get(&e).copied())
    }

    /// Parses the TOML form (see the type-level docs).
    ///
    /// # Errors
    /// [`Error::InvalidInput`] describing the first problem found.
    #[cfg(feature = "radii-file")]
    pub fn from_toml_str(text: &str) -> Result<Self> {
        use std::collections::BTreeMap;

        #[derive(serde::Deserialize)]
        #[serde(deny_unknown_fields)]
        struct File {
            base: Option<String>,
            fallback: Option<f64>,
            #[serde(default)]
            elements: BTreeMap<String, f64>,
            #[serde(default)]
            atoms: BTreeMap<String, f64>,
        }

        let file: File =
            toml::from_str(text).map_err(|e| Error::InvalidInput(format!("radius file: {e}")))?;
        let base = match file.base.as_deref().map(str::to_ascii_lowercase).as_deref() {
            None | Some("bondi") => Some(BuiltinRadii::Bondi),
            Some("alvarez") => Some(BuiltinRadii::Alvarez),
            Some("protor") => Some(BuiltinRadii::ProtOr),
            Some("none") => None,
            Some(other) => {
                return Err(Error::InvalidInput(format!(
                    "radius file: unknown base set '{other}' (bondi, alvarez, protor, none)"
                )));
            }
        };
        let mut rules = Self::new(base);
        if let Some(f) = file.fallback {
            rules.set_fallback(f)?;
        }
        for (symbol, r) in file.elements {
            let e = Element::from_symbol(&symbol).ok_or_else(|| {
                Error::InvalidInput(format!("radius file: unknown element '{symbol}'"))
            })?;
            rules.set_element(e, r)?;
        }
        for (key, r) in file.atoms {
            let (res, atom) = key.split_once(':').ok_or_else(|| {
                Error::InvalidInput(format!(
                    "radius file: atom key '{key}' must be RESIDUE:ATOM (use * as wildcard)"
                ))
            })?;
            rules.set_atom(res, atom, r)?;
        }
        Ok(rules)
    }

    /// Reads [`RadiusRules::from_toml_str`] from a file.
    ///
    /// # Errors
    /// [`Error::Io`] or [`Error::InvalidInput`].
    #[cfg(feature = "radii-file")]
    pub fn from_path(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path).map_err(|source| Error::Io {
            path: Some(path.to_path_buf()),
            source,
        })?;
        Self::from_toml_str(&text).map_err(|e| match e {
            Error::InvalidInput(m) => Error::InvalidInput(format!("{}: {m}", path.display())),
            other => other,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn precedence() {
        let mut r = RadiusRules::new(None);
        r.set_element(Element::C, 1.0).unwrap();
        r.set_atom("*", "CA", 2.0).unwrap();
        r.set_atom("gly", "*", 3.0).unwrap();
        r.set_atom("GLY", "CA", 4.0).unwrap();
        assert_eq!(r.lookup_rules("GLY", "ca", Some(Element::C)), Some(4.0));
        assert_eq!(r.lookup_rules("ALA", "CA", Some(Element::C)), Some(2.0));
        assert_eq!(r.lookup_rules("GLY", "CB", Some(Element::C)), Some(3.0));
        assert_eq!(r.lookup_rules("ALA", "CB", Some(Element::C)), Some(1.0));
        assert_eq!(r.lookup_rules("ALA", "N", Some(Element::N)), None);
        assert!(r.set_element(Element::C, -1.0).is_err());
        assert!(r.set_atom("", "CA", 1.0).is_err());
        assert!(r.set_fallback(f64::NAN).is_err());
    }

    #[cfg(feature = "radii-file")]
    #[test]
    fn toml_file() {
        let r = RadiusRules::from_toml_str(
            "base = \"ProtOr\"\nfallback = 1.9\n[elements]\nFe = 2.0\n[atoms]\n\"HEM:FE\" = 1.6\n",
        )
        .unwrap();
        assert_eq!(r.base(), Some(BuiltinRadii::ProtOr));
        assert_eq!(r.fallback(), Some(1.9));
        assert_eq!(r.lookup_rules("HEM", "FE", Some(Element::Fe)), Some(1.6));
        assert_eq!(r.lookup_rules("XYZ", "FE", Some(Element::Fe)), Some(2.0));
        for bad in [
            "base = \"nope\"",
            "[elements]\nQq = 1.0",
            "[atoms]\nCA = 1.0",
            "[atoms]\n\"A:B\" = -2",
            "colour = 3",
        ] {
            assert!(RadiusRules::from_toml_str(bad).is_err(), "{bad}");
        }
        assert_eq!(
            RadiusRules::from_toml_str("").unwrap().base(),
            Some(BuiltinRadii::Bondi)
        );
    }
}
