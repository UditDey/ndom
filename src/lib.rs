use std::thread;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{sync_channel, SyncSender};

mod renderer;
mod layout;

pub mod html;
pub mod css;

use winit::{
    window::{Window, WindowId, WindowAttributes},
    event_loop::{EventLoop, ActiveEventLoop, ControlFlow},
    event::WindowEvent,
    application::ApplicationHandler
};

use slotmap::{DefaultKey, SecondaryMap};
use taffy::{Style, Size, FlexDirection};

use crate::{renderer::Renderer, css::Color, layout::LayoutTree};

pub(crate) struct Box {
    radius: f32,
    background_color: Color
}

pub struct Dom {
    layout_tree: LayoutTree,
    root_node: taffy::NodeId,
    box_list: SecondaryMap<DefaultKey, Box>
}

impl Dom {
    pub fn new() -> Self {
        // Setup the layout tree
        let mut layout_tree = LayoutTree::new();

        // Root node style that occupies all the available window space
        // and lays nodes vertically downwards
        let root_style = Style {
            size: Size::from_percent(1.0, 1.0),
            flex_direction: FlexDirection::Column,
            ..Default::default()
        };

        let root_node = layout_tree.new_leaf(root_style);

        Self {
            layout_tree,
            root_node,
            box_list: SecondaryMap::new()
        }
    }
}

pub enum AppEvent<'a> {
    Init(&'a mut Dom)
}

pub fn run_app(handler: impl Fn(AppEvent)) {
    // Setup winit
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Wait);

    // Setup the DOM
    let mut dom = Dom::new();

    // Process the app init event
    handler(AppEvent::Init(&mut dom));

    // The DOM will be shared between window and render threads
    let dom = Mutex::new(dom);
    let should_exit = AtomicBool::new(false);

    // Since the renderer is created on the window thread we use this
    // channel to send it over to the render thread
    // Winit's new event loop model (starting from 0.30.0) makes this
    // all...pretty convoluted and this is the most reasonable approach
    // to do this from what I can tell
    let (tx, rx) = sync_channel::<Renderer>(1);

    thread::scope(|s| {
        // Run the render loop on another thread
        let render_thread = s.spawn({
            let dom = &dom;
            let should_exit = &should_exit;

            move || {
                // Wait for us to recieve the renderer
                let mut renderer = rx.recv().unwrap();

                renderer.run(dom, || should_exit.load(Ordering::SeqCst));
            }
        });

        // Run the window loop on the main thread
        struct WinitHandler<'a> {
            window: Option<Window>,
            dom: &'a Mutex<Dom>,
            tx: SyncSender<Renderer>,
            should_exit: &'a AtomicBool
        }

        impl<'a> ApplicationHandler for WinitHandler<'a> {
            fn resumed(&mut self, event_loop: &ActiveEventLoop) {
                // If the window hasn't been created yet, create it along with the renderer
                // and send it to the render thread
                if self.window.is_none() {
                    let attr = WindowAttributes::default().with_visible(false);

                    let window = event_loop.create_window(attr).unwrap();
                    let renderer = Renderer::new(&window);

                    self.tx.send(renderer).unwrap();

                    window.set_visible(true);
                    self.window = Some(window);
                }
            }

            fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
                if let WindowEvent::CloseRequested = event {
                    self.should_exit.store(true, Ordering::SeqCst);
                    event_loop.exit();
                }
            }
        }

        let mut winit_handler = WinitHandler {
            window: None,
            dom: &dom,
            tx,
            should_exit: &should_exit
        };

        event_loop.run_app(&mut winit_handler).unwrap();

        // Wait for render loop to exit before dropping the window
        render_thread.join().unwrap();
    });
}
