use crate::RAG;
use html5ever::interface::TreeSink;
use shared::{END_OF_SENTENCE, SPLIT_WORD};
use std::io::Write;

impl RAG {
    pub fn parse_website(&self, url: &str) {
        let site_string = ureq::get(url).call().unwrap().into_string().unwrap();

        let mut site = scraper::Html::parse_document(&site_string);

        let has_main = site_string.contains("</main>");

        if has_main {
            let main_id = site
                .root_element()
                .descendants()
                .find(|node| match node.value() {
                    scraper::Node::Element(element) => element.name() == "main",
                    _ => false,
                })
                .unwrap()
                .id();

            site.tree.get_mut(main_id).unwrap().detach();

            let body_id = site
                .root_element()
                .children()
                .find(|node| match node.value() {
                    scraper::Node::Element(element) => element.name() == "body",
                    _ => false,
                })
                .unwrap()
                .id();

            let ids_to_delete = site
                .tree
                .get(body_id)
                .unwrap()
                .children()
                .map(|node| node.id())
                .collect::<Vec<_>>();

            site.tree.get_mut(body_id).unwrap().append_id(main_id);

            for id in ids_to_delete {
                site.remove_from_parent(&id);
            }
        }

        let nodes_to_delete = site
            .root_element()
            .descendants()
            .filter_map(|node| {
                let id = node.id();
                match node.value() {
                    scraper::Node::Element(element) => match element.name() {
                        "link" | "meta" | "script" | "cite" | "footer" | "style" | "figcaption"
                        | "noscript" | "figure" => Some(id),
                        _ => node
                            .descendants()
                            .all(|node| {
                                !node.value().is_text()
                                    || node.value().as_text().unwrap().trim().is_empty()
                            })
                            .then(|| id),
                    },
                    scraper::Node::Comment(_) | scraper::Node::Doctype(_) => Some(id),
                    _ => None,
                }
            })
            .collect::<Vec<_>>();

        for node_id in nodes_to_delete.into_iter() {
            site.remove_from_parent(&node_id);
        }

        for node in site.tree.values_mut() {
            match node {
                scraper::Node::Element(element) => {
                    element.attrs.clear();
                }
                _ => {}
            }
        }

        let stripped = site.html();

        let mut stripped = stripped
            .lines()
            .flat_map(|line| [line.trim(), "\n"])
            .collect::<String>();

        let mut prev = 'a';
        stripped.retain(|c| {
            let should_keep = c != '\n' || prev != '\n';
            prev = c;
            should_keep
        });

        let mut file = std::fs::File::create("stripped.html").unwrap();
        file.write_all(stripped.as_bytes()).unwrap();

        let md = mdka::from_html(&stripped);

        let mut md = md
            .lines()
            .flat_map(|line| {
                let trimmed = line.trim();

                let is_empty = trimmed
                    .trim_matches(|c: char| {
                        SPLIT_WORD.contains(&c)
                            || END_OF_SENTENCE.contains(&c)
                            || c == '-'
                            || c == '|'
                    })
                    .is_empty();

                // "!trimmed.is_empty()" is there to keep empty lines
                // without it all paragraphs get glued together
                if !trimmed.is_empty() && is_empty {
                    ["", ""]
                } else {
                    [trimmed, "\n"]
                }
            })
            .collect::<String>();

        let mut prev = ['a'; 2];
        md.retain(|c| {
            let should_keep = c != '\n' || prev != ['\n', '\n'];
            prev[0] = prev[1];
            prev[1] = c;
            should_keep
        });

        let mut file = std::fs::File::create("./stripped.md").unwrap();
        file.write_all(md.as_bytes()).unwrap();
    }
}
