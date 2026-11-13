// sesmesh — Solvent Excluded Surface meshes for molecules
// Copyright (c) 2026 Ugo Varetto
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared model bookkeeping for formats with MODEL/ENDMDL-style records.

use crate::structure::{AtomRecord, Model};

/// Collects atoms into models. Atoms outside any `MODEL` record go into an
/// implicit model numbered after the previous one (1 for the first).
#[derive(Debug, Default)]
pub(crate) struct ModelBuilder {
    models: Vec<Model>,
    current: Option<Model>,
}

impl ModelBuilder {
    /// Starts a new model (closing any open one). `None` numbers it after the
    /// previous model.
    pub(crate) fn begin(&mut self, number: Option<i32>) {
        self.end();
        let number = number.unwrap_or_else(|| self.next_number());
        self.current = Some(Model::new(number));
    }

    /// Closes the open model, if any. Empty models are dropped.
    pub(crate) fn end(&mut self) {
        if let Some(model) = self.current.take()
            && !model.atoms.is_empty()
        {
            self.models.push(model);
        }
    }

    /// Adds an atom to the open model, opening an implicit one if needed.
    pub(crate) fn push(&mut self, atom: AtomRecord) {
        if self.current.is_none() {
            self.current = Some(Model::new(self.next_number()));
        }
        self.current
            .as_mut()
            .expect("model opened above")
            .atoms
            .push(atom);
    }

    /// Closes the open model and returns all non-empty models.
    pub(crate) fn finish(mut self) -> Vec<Model> {
        self.end();
        self.models
    }

    fn next_number(&self) -> i32 {
        self.models.last().map_or(1, |m| m.number.saturating_add(1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn implicit_and_explicit_models() {
        let mut b = ModelBuilder::default();
        b.push(AtomRecord::new([0.0; 3]));
        b.begin(Some(5));
        b.end();
        b.begin(Some(7));
        b.push(AtomRecord::new([1.0; 3]));
        b.end();
        b.push(AtomRecord::new([2.0; 3]));
        let models = b.finish();
        let numbers: Vec<i32> = models.iter().map(|m| m.number).collect();
        assert_eq!(numbers, [1, 7, 8]);
    }
}
