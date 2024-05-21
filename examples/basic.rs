use ndom::{AppEvent, html::Div, css::{StyleRule, Length, Color}};

fn main() {
    ndom::run_app(|event| {
        match event {
            // App init event, setup the DOM here
            AppEvent::Init(dom) => {
                // Setting up style rules
                let style_1 = StyleRule::new()
                    .padding(Length::Px(20.0))
                    .background_color(Color::MAROON);

                let style_2 = StyleRule::new()
                    .padding(Length::Px(20.0))
                    .background_color(Color::AQUA);

                let style_3 = StyleRule::new()
                    .padding(Length::Px(15.0))
                    .background_color(Color::FUCHSIA)
                    .border_radius(10.0);

                // Adding `div` elements with inline styles
                Div::new()
                    .style(style_1)
                    .add(dom);

                let elem = Div::new()
                    .style(style_2)
                    .add(dom);

                Div::new()
                    .style(style_3)
                    .add_into(elem, dom);
            }
        }
    });
}
