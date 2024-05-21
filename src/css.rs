use taffy::{Style as TaffyStyle, Rect, Size, FlexDirection};

use crate::{Dom, Box, html::ElemKind};

#[derive(Debug, Clone, Copy)]
pub enum Length {
    Px(f32)
}

#[derive(Debug, Clone, Copy)]
pub enum Color {
    Rgb(u8, u8, u8),
    Rgba(u8, u8, u8, u8)
}

impl Color {
    // CSS standard colors, taken from https://developer.mozilla.org/en-US/docs/Web/CSS/named-color
    pub const BLACK:   Self =  Self::Rgb(0x00, 0x00, 0x00);
    pub const SILVER:  Self =  Self::Rgb(0xC0, 0xC0, 0xC0);
    pub const GRAY:    Self =  Self::Rgb(0x80, 0x80, 0x80);
    pub const WHITE:   Self =  Self::Rgb(0xFF, 0xFF, 0xFF);
    pub const MAROON:  Self =  Self::Rgb(0x80, 0x00, 0x00);
    pub const RED:     Self =  Self::Rgb(0xFF, 0x00, 0x00);
    pub const PURPLE:  Self =  Self::Rgb(0x80, 0x00, 0x80);
    pub const FUCHSIA: Self =  Self::Rgb(0xFF, 0x00, 0xFF);
    pub const GREEN:   Self =  Self::Rgb(0x00, 0x80, 0x00);
    pub const LIME:    Self =  Self::Rgb(0x00, 0xFF, 0x00);
    pub const OLIVE:   Self =  Self::Rgb(0x80, 0x80, 0x00);
    pub const YELLOW:  Self =  Self::Rgb(0xFF, 0xFF, 0x00);
    pub const NAVY:    Self =  Self::Rgb(0x00, 0x00, 0x80);
    pub const BLUE:    Self =  Self::Rgb(0x00, 0x00, 0xFF);
    pub const TEAL:    Self =  Self::Rgb(0xFF, 0x80, 0x80);
    pub const AQUA:    Self =  Self::Rgb(0x00, 0xFF, 0xFF);

    pub(crate) fn as_f32_array(&self) -> [f32; 4] {
        match self {
            Self::Rgb(r, g, b) => {
                let r = *r as f32 / 255.0;
                let g = *g as f32 / 255.0;
                let b = *b as f32 / 255.0;

                [r, g, b, 1.0]
            },

            Self::Rgba(r, g, b, a) => {
                let r = *r as f32 / 255.0;
                let g = *g as f32 / 255.0;
                let b = *b as f32 / 255.0;
                let a = *a as f32 / 255.0;

                [r, g, b, a]
            }
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct StyleRule {
    pub(crate) elem: Option<ElemKind>,
    pub(crate) class: Option<String>,
    pub(crate) id: Option<String>,
    pub(crate) padding: Option<Length>,
    pub(crate) background_color: Option<Color>,
    pub(crate) border_radius: Option<f32>
}

impl StyleRule {
    #[must_use = "This `StyleRule` will not registered with the DOM tree unless `add()` is called"]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use = "This `StyleRule` will not registered with the DOM tree unless `add()` is called"]
    pub fn select_elem(mut self, elem: ElemKind) -> Self {
        self.elem = Some(elem);
        self
    }

    #[must_use = "This `StyleRule` will not registered with the DOM tree unless `add()` is called"]
    pub fn select_class(mut self, class: String) -> Self {
        self.class = Some(class);
        self
    }

    #[must_use = "This `StyleRule` will not registered with the DOM tree unless `add()` is called"]
    pub fn select_id(mut self, id: String) -> Self {
        self.id = Some(id);
        self
    }

    #[must_use = "This `StyleRule` will not registered with the DOM tree unless `add()` is called"]
    pub fn padding(mut self, padding: Length) -> Self {
        self.padding = Some(padding);
        self
    }

    #[must_use = "This `StyleRule` will not registered with the DOM tree unless `add()` is called"]
    pub fn background_color(mut self, color: Color) -> Self {
        self.background_color = Some(color);
        self
    }

    #[must_use = "This `StyleRule` will not registered with the DOM tree unless `add()` is called"]
    pub fn border_radius(mut self, radius: f32) -> Self {
        self.border_radius = Some(radius);
        self
    }

    pub fn add(self, _dom: &mut Dom) {
        todo!()
    }

    pub(crate) fn as_taffy_style(&self) -> TaffyStyle {
        let mut style = TaffyStyle {
            flex_direction: FlexDirection::Column,
            ..Default::default()
        };

        if let Some(padding) = self.padding {
            style.padding = match padding {
                Length::Px(px) => Rect::length(px)
            };
        }

        style
    }

    pub(crate) fn as_dom_box(&self) -> Box {
        Box {
            radius: self.border_radius.unwrap_or(0.0),
            background_color: self.background_color.unwrap_or(Color::WHITE)
        }
    }
}
