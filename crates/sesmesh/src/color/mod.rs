//! Vertex colour palettes for mesh output (Spec FR-OUT-2).
//!
//! Colours are linear `[r, g, b]` in `0..=1`. The default element palette is
//! the project's `atom-colors.json` (118 elements); categorical colours serve
//! chains and components.
//!
//! ```
//! use sesmesh::color::{ElementPalette, parse_hex};
//! use sesmesh::Element;
//!
//! let mut palette = ElementPalette::default();
//! let carbon = palette.color(Element::from_symbol("C"));
//! assert_eq!(carbon, parse_hex("#535b64").unwrap());
//! palette.set(Element::from_symbol("C").unwrap(), [0.5, 0.5, 0.5]);
//! assert_eq!(palette.color(Element::from_symbol("C")), [0.5, 0.5, 0.5]);
//! ```

mod element_table;

use crate::mesh::PatchKind;
use crate::structure::Element;

/// An RGB colour with components in `0..=1`.
pub type Rgb = [f32; 3];

/// Parses `#rrggbb` (the `#` is optional, case-insensitive).
pub fn parse_hex(text: &str) -> Option<Rgb> {
    let h = text.trim().trim_start_matches('#');
    if h.len() != 6 || !h.is_ascii() {
        return None;
    }
    let c = |i: usize| u8::from_str_radix(&h[i..i + 2], 16).ok();
    Some([c(0)?, c(2)?, c(4)?].map(|v| f32::from(v) / 255.0))
}

/// Per-element colours.
#[derive(Debug, Clone, PartialEq)]
pub struct ElementPalette {
    /// Index Z − 1.
    colors: Vec<Rgb>,
    unknown: Rgb,
}

impl Default for ElementPalette {
    /// The bundled `atom-colors.json` palette; unknown elements are grey.
    fn default() -> Self {
        Self {
            colors: element_table::ELEMENT_RGB
                .iter()
                .map(|c| c.map(|v| f32::from(v) / 255.0))
                .collect(),
            unknown: [0.6, 0.6, 0.6],
        }
    }
}

impl ElementPalette {
    /// Colour of `element` (`None`: the unknown colour).
    pub fn color(&self, element: Option<Element>) -> Rgb {
        element
            .and_then(|e| self.colors.get(usize::from(e.atomic_number()) - 1))
            .copied()
            .unwrap_or(self.unknown)
    }

    /// Overrides the colour of `element`.
    pub fn set(&mut self, element: Element, rgb: Rgb) {
        if let Some(c) = self
            .colors
            .get_mut(usize::from(element.atomic_number()) - 1)
        {
            *c = rgb;
        }
    }

    /// Overrides the colour of atoms without a known element.
    pub fn set_unknown(&mut self, rgb: Rgb) {
        self.unknown = rgb;
    }
}

/// Ten well-separated colours (Tableau 10) for chains and components.
pub const CATEGORICAL: [Rgb; 10] = [
    [0.122, 0.467, 0.706],
    [1.000, 0.498, 0.055],
    [0.173, 0.627, 0.173],
    [0.839, 0.153, 0.157],
    [0.580, 0.404, 0.741],
    [0.549, 0.337, 0.294],
    [0.890, 0.467, 0.761],
    [0.498, 0.498, 0.498],
    [0.737, 0.741, 0.133],
    [0.090, 0.745, 0.812],
];

/// Categorical colour `i` (cycling).
pub fn categorical(i: usize) -> Rgb {
    CATEGORICAL[i % CATEGORICAL.len()]
}

/// Colour of a patch kind: convex light blue, saddle green, concave orange.
pub fn patch_color(patch: PatchKind) -> Rgb {
    match patch {
        PatchKind::Convex => [0.55, 0.71, 0.92],
        PatchKind::Saddle => [0.40, 0.76, 0.40],
        PatchKind::Concave => [0.96, 0.60, 0.25],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hex_parsing() {
        assert_eq!(parse_hex("#ff0000"), Some([1.0, 0.0, 0.0]));
        assert_eq!(parse_hex("00FF00"), Some([0.0, 1.0, 0.0]));
        for bad in ["#fff", "#gg0000", "", "#ff00000", "#ff00é"] {
            assert_eq!(parse_hex(bad), None, "{bad}");
        }
    }

    /// The generated table equals the bundled JSON file.
    #[test]
    fn table_matches_json() {
        let text = include_str!("../../data/atom-colors.json");
        let json: serde_json::Value = serde_json::from_str(text).unwrap();
        assert_eq!(json["format"], "eigenvista-atom-colors");
        let colors = json["colors"].as_array().unwrap();
        assert_eq!(colors.len(), 118);
        let palette = ElementPalette::default();
        for c in colors {
            let z = u8::try_from(c["z"].as_u64().unwrap()).unwrap();
            let e = Element::from_atomic_number(z).unwrap();
            assert_eq!(e.symbol(), c["symbol"].as_str().unwrap());
            let want = parse_hex(c["color"].as_str().unwrap()).unwrap();
            assert_eq!(palette.color(Some(e)), want, "{}", e.symbol());
        }
        assert_eq!(palette.color(None), [0.6, 0.6, 0.6]);
    }

    #[test]
    fn categorical_cycles() {
        assert_eq!(categorical(0), categorical(10));
        assert_ne!(categorical(0), categorical(1));
        assert_ne!(
            patch_color(PatchKind::Saddle),
            patch_color(PatchKind::Concave)
        );
    }
}
