// sesmesh — Solvent Excluded Surface meshes for molecules
// Copyright (c) 2026 Ugo Varetto
// SPDX-License-Identifier: MIT OR Apache-2.0

//! String interning for atom, residue and chain names.

use rustc_hash::FxHashMap;

/// Handle to a string stored in a [`StringPool`].
///
/// `Sym::default()` is the empty string.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Sym(u32);

impl Sym {
    /// The empty string.
    pub const EMPTY: Sym = Sym(0);

    /// Index of the string inside its pool.
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

/// Interned strings of a [`Structure`](super::Structure).
///
/// Atom, residue and chain names repeat massively, so atom records store
/// [`Sym`] handles and the pool stores each distinct string once.
#[derive(Debug, Clone)]
pub struct StringPool {
    strings: Vec<Box<str>>,
    index: FxHashMap<Box<str>, Sym>,
}

impl Default for StringPool {
    fn default() -> Self {
        let mut pool = Self {
            strings: Vec::new(),
            index: FxHashMap::default(),
        };
        pool.intern("");
        pool
    }
}

impl StringPool {
    /// Creates a pool containing only the empty string.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the handle for `s`, adding it if needed.
    ///
    /// # Panics
    /// Beyond 2³² distinct strings.
    pub fn intern(&mut self, s: &str) -> Sym {
        if let Some(&sym) = self.index.get(s) {
            return sym;
        }
        let sym = Sym(u32::try_from(self.strings.len()).expect("fewer than 2^32 distinct strings"));
        self.strings.push(s.into());
        self.index.insert(s.into(), sym);
        sym
    }

    /// Interns raw bytes; invalid UTF-8 is replaced lossily.
    pub(crate) fn intern_bytes(&mut self, bytes: &[u8]) -> Sym {
        match std::str::from_utf8(bytes) {
            Ok(s) => self.intern(s),
            Err(_) => self.intern(&String::from_utf8_lossy(bytes)),
        }
    }

    /// The string behind `sym`.
    ///
    /// # Panics
    /// If `sym` comes from a different pool and is out of range.
    pub fn get(&self, sym: Sym) -> &str {
        &self.strings[sym.index()]
    }

    /// The handle of `s`, if it was interned.
    pub fn find(&self, s: &str) -> Option<Sym> {
        self.index.get(s).copied()
    }

    /// Number of distinct strings (including the empty string).
    pub fn len(&self) -> usize {
        self.strings.len()
    }

    /// Always `false`: the pool contains at least the empty string.
    pub fn is_empty(&self) -> bool {
        false
    }

    /// Iterates over `(handle, string)` pairs in insertion order.
    pub fn iter(&self) -> impl Iterator<Item = (Sym, &str)> {
        self.strings
            .iter()
            .enumerate()
            .map(|(i, s)| (Sym(i as u32), &**s))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interning_is_stable() {
        let mut pool = StringPool::new();
        assert_eq!(pool.get(Sym::EMPTY), "");
        assert_eq!(pool.intern(""), Sym::EMPTY);
        let a = pool.intern("ALA");
        let b = pool.intern("GLY");
        assert_eq!(pool.intern("ALA"), a);
        assert_ne!(a, b);
        assert_eq!(pool.get(b), "GLY");
        assert_eq!(pool.find("GLY"), Some(b));
        assert_eq!(pool.find("SER"), None);
        assert_eq!(pool.len(), 3);
        assert_eq!(pool.intern_bytes(b"\xffX"), pool.find("\u{fffd}X").unwrap());
    }
}
