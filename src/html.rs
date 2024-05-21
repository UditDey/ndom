use taffy::{Style as TaffyStyle, NodeId};

use crate::{Dom, Box, css::{StyleRule, Color}};

#[derive(Debug, Clone, Copy)]
pub enum ElemKind {
    Div
}

pub struct Elem(NodeId);

pub struct Div {
    id: Option<String>,
    class: Option<String>,
    inline_style: Option<StyleRule>
}

impl Div {
    #[must_use = "This `div` will not placed into the DOM tree unless `add()` or `add_into()` is called"]
    pub fn new() -> Self {
        Self {
            id: None,
            class: None,
            inline_style: None
        }
    }

    #[must_use = "This `div` will not placed into the DOM tree unless `add()` or `add_into()` is called"]
    pub fn id(mut self, id: String) -> Self {
        self.id = Some(id);
        self
    }

    #[must_use = "This `div` will not placed into the DOM tree unless `add()` or `add_into()` is called"]
    pub fn class(mut self, class: String) -> Self {
        self.class = Some(class);
        self
    }

    #[must_use = "This `div` will not placed into the DOM tree unless `add()` or `add_into()` is called"]
    pub fn style(mut self, style: StyleRule) -> Self {
        self.inline_style = Some(style);
        self
    }

    pub fn add(self, dom: &mut Dom) -> Elem {
        self.add_inner(dom, dom.root_node)
    }

    pub fn add_into(self, parent: Elem, dom: &mut Dom) -> Elem {
        self.add_inner(dom, parent.0)
    }

    fn add_inner(self, dom: &mut Dom, root_node: NodeId) -> Elem {
        let taffy_style = match &self.inline_style {
            Some(inline_style) => inline_style.as_taffy_style(),
            None => TaffyStyle::default()
        };

        // Make a new taffy tree node for this div
        let node = dom.layout_tree.new_leaf(taffy_style);
        dom.layout_tree.add_child(root_node, node);

        // Make an entry in the box list
        let div_box = match &self.inline_style {
            Some(inline_style) => inline_style.as_dom_box(),
            None => todo!()
        };

        dom.box_list.insert(node.into(), div_box);

        Elem(node)
    }
}
